"""Coverage + accuracy report for the MID mapping produced by map_paths_to_mid.py.

Accuracy is measured on an independent signal that the mapper never sees: the gold evidence triples
(data/gold_paths, which carry MIDs). A predicted triple is comparable when its name-level form
(subject surface, predicate, object surface) equals a gold triple's (subject name, predicate, object name);
it is then correct iff the resolved (subject MID, object MID) equals the gold MIDs.
The same comparable triples are re-mapped with naive 1:1 name lookup as a baseline:
  - naive_first  : first MID returned for the exact English name (what a name->MID dict does)
  - naive_degree : the most connected MID among the exact-name matches (popularity prior)

Usage:
  python src/mid_mapping/evaluate_mid_mapping.py
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

from map_paths_to_mid import DATASETS, METHODS, MID_RE, NAME, KG, Sparql, lit, norm


def gold_index(path):
    idx = defaultdict(dict)  # qid -> name-level key -> set of (s_id, o_id)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qid = re.sub(r"\.P\d+$", "", r["id"])
            for ge in r.get("gold_evidence") or []:
                for t in ge["evidence_triples"]:
                    s, o = t["subject"], t["object"]
                    if s["kind"] != "entity" or o["kind"] != "entity":
                        continue
                    key = (norm(s.get("name") or s["id"]), t["predicate"]["id"], norm(o.get("name") or o["id"]))
                    idx[qid].setdefault(key, set()).add((s["id"], o["id"]))
    return idx


def iter_triples(rec):
    for it in rec["items"]:
        for t in it.get("triples", [it]):
            yield it, t


def naive_mid(kg, surface, how):
    if MID_RE.match(surface):
        return surface, 1
    q = f"SELECT DISTINCT ?e WHERE {{ ?e {NAME} {lit(surface)}@en }} LIMIT 200"
    mids = [b["e"]["id"] for b in kg.sp.select(q) if b["e"]["kind"] == "entity"]
    if not mids:
        return None, 0
    if how == "first":
        return mids[0], len(mids)
    return max(mids, key=kg.degree), len(mids)


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.join(root, "data"))
    p.add_argument("--mid_dir", default=os.path.join(root, "data", "mid_paths"))
    p.add_argument("--endpoint", default="http://localhost:3001/sparql")
    p.add_argument("--cache", default=os.path.join(root, "data", "mid_paths", ".cache", "sparql_cache.sqlite"))
    args = p.parse_args()

    sparql = Sparql(args.endpoint, args.cache)
    kg = KG(sparql)
    report = {}
    for ds in DATASETS:
        gold = gold_index(os.path.join(args.data_dir, "gold_paths", f"{ds}.gold.jsonl"))
        for m in METHODS:
            path = os.path.join(args.mid_dir, m, f"{ds}.{m}.mid.jsonl")
            if not os.path.exists(path):
                continue
            item_status, triple_status, resolution, pick, start = Counter(), Counter(), Counter(), Counter(), Counter()
            acc = Counter()
            n_rec = 0
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    n_rec += 1
                    g = gold.get(rec["id"], {})
                    for it in rec["items"]:
                        item_status[it["status"]] += 1
                        if "start" in it:
                            start[it["start"]] += 1
                    for it, t in iter_triples(rec):
                        triple_status[t["status"]] += 1
                        resolution[t.get("resolution", "path_chain")] += 1
                        if "pick" in t:
                            pick[t["pick"]] += 1
                        if t["status"] not in ("resolved", "ambiguous"):
                            continue
                        s, o = t["subject"], t["object"]
                        if s.get("kind") != "entity" or o.get("kind") != "entity":
                            continue
                        key = (norm(s["surface"]), t["predicate"], norm(o["surface"]))
                        if key not in g:
                            continue
                        golds = g[key]
                        acc["comparable"] += 1
                        acc[f"comparable_{t['status']}"] += 1
                        ok = (s["id"], o["id"]) in golds
                        mode = t.get("resolution", "path_chain")
                        acc["ours_correct"] += ok
                        acc[f"ours_correct_{t['status']}"] += ok
                        acc[f"comparable_mode_{mode}"] += 1
                        acc[f"ours_correct_mode_{mode}"] += ok
                        if t["status"] == "ambiguous" and "candidates" in t:
                            acc["ambiguous_gold_in_candidates"] += any(tuple(c) in golds for c in t["candidates"])
                        for how in ("first", "degree"):
                            ns, ks = naive_mid(kg, s["surface"], how)
                            no, ko = naive_mid(kg, o["surface"], how)
                            acc[f"naive_{how}_correct"] += (ns, no) in golds
                            if how == "first":
                                acc["naive_name_ambiguous"] += (ks > 1 or ko > 1)
            n_tr = sum(triple_status.values())
            report[f"{ds}/{m}"] = {
                "records": n_rec,
                "items": dict(item_status),
                "path_start": dict(start),
                "triples": dict(triple_status),
                "triple_resolved_rate": round(triple_status["resolved"] / max(n_tr, 1), 4),
                "triple_grounded_rate": round((triple_status["resolved"] + triple_status["ambiguous"]) / max(n_tr, 1), 4),
                "resolution": dict(resolution),
                "ambiguity_pick": dict(pick),
                "gold_check": dict(acc),
            }
            c = max(acc["comparable"], 1)
            print(f"{ds:7s} {m:12s} triples={n_tr:7d} resolved={report[f'{ds}/{m}']['triple_resolved_rate']:.3f} "
                  f"grounded={report[f'{ds}/{m}']['triple_grounded_rate']:.3f} | gold-comparable={acc['comparable']:6d} "
                  f"ours={acc['ours_correct'] / c:.3f} naive_first={acc['naive_first_correct'] / c:.3f} "
                  f"naive_degree={acc['naive_degree_correct'] / c:.3f} name_ambiguous={acc['naive_name_ambiguous'] / c:.3f}")
    sparql.close()
    out = os.path.join(args.mid_dir, "mapping_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("->", out)


if __name__ == "__main__":
    main()
