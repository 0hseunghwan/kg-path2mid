"""Map name-level reasoning paths / triples of KGQA methods to Freebase MIDs.

Supported inputs (data/<method>_paths/<dataset>.<method>.jsonl):
  - rog, gnnrag      : "Reasoning Paths:" block, one path per line  (e0 -> r1 -> e1 -> r2 -> e2 ...)
  - subgraphrag      : "Triplets:" block in `user_query`, one (h,r,t) per line
  - parallaxrag      : "Triplets:" block in `input`, one (h,r,t) per line

Name -> MID is NOT done by 1:1 lookup (more than half of the entity names in these files map to several MIDs:
films / soundtracks / releases / people sharing a name). Instead every name is grounded *through an edge*:

  1. Seeds: question entities = MID constants in the dataset SPARQL (data/gold_paths). Answer MIDs are never used.
  2. Paths (RoG, GNN-RAG): beam search from the seed along the path. Hop i keeps only the MIDs b that satisfy
     (a, r_i, b) or (b, r_i, a) in Freebase for some surviving a, AND name(b) == e_{i+1}. Chains that die later are
     pruned backwards, so a node is only kept if a full chain through it exists.
  3. Triples (SubgraphRAG, ParallaxRAG): propagate from the seeds through the retrieved subgraph. A triple with an
     already-grounded endpoint is resolved by an anchored edge query; newly grounded endpoints become anchors.
     Triples never reached from a seed are resolved by a relation-constrained query on both names
     (?h name H . ?h r ?t . ?t name T), which removes most homonyms by type alone.
  4. Remaining ties are flagged `ambiguous` and keep every candidate; the primary pick prefers MIDs already grounded
     elsewhere in the same sample, then higher KG degree.

Usage:
  python src/mid_mapping/map_paths_to_mid.py --method rog --dataset WebQSP
  python src/mid_mapping/map_paths_to_mid.py --method all --dataset all --workers 8
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

NS = "http://rdf.freebase.com/ns/" # URI 가장 흔한 타입을 디폴트 설정
NAME = f"<{NS}type.object.name>"
ALIAS = f"<{NS}common.topic.alias>"
METHODS = ["rog", "gnnrag", "subgraphrag", "parallaxrag"]
DATASETS = ["WebQSP", "cwq"]

MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$") # m 또는 g 로 시작 + . + 0~9,a~z,_ 로만 구성
REL_RE = re.compile(r"^[A-Za-z0-9_\-]+([.#][A-Za-z0-9_\-]+)+$")
TRIPLE_RE = re.compile(r",([A-Za-z0-9_\-]+(?:[.#][A-Za-z0-9_\-]+)+),")
MAX_ANCHORS = 200  # VALUES size per query
MAX_CHAINS = 50
RANK = {"mids": 0, "mid": 0, "name": 0, "literal": 1, "alias": 2, "name_ci": 3}
SELECTIVE = {"mids", "mid", "name", "alias"}


# URI 는 resource 에 대응되는 링크라고 생각하면 됨. object type 중에 uri/literal 있는데
# literal 은 1992-02-01 같은 직접적인 값을 의미하는 듯

# Freebase relation(predicate)의 URI 형태는 rdf.freebase.com 외에 www.w3.org 형태(메타 데이터 저장용)도 존재
# 전체 uri 없고 일부만 존재하는 예외 케이스 처리 코드 -> predicate URI 정규화
def rel_iri(rel):
    if rel.startswith("rdf-schema#"):
        return "<http://www.w3.org/2000/01/rdf-schema#" + rel.split("#", 1)[1] + ">"
    if rel.startswith("22-rdf-syntax-ns#"):
        return "<http://www.w3.org/1999/02/22-rdf-syntax-ns#" + rel.split("#", 1)[1] + ">"
    if rel.startswith("owl#"):
        return "<http://www.w3.org/2002/07/owl#" + rel.split("#", 1)[1] + ">"
    return f"<{NS}{rel}>"

# MID 예외 케이스 처리
def mid_iri(mid):
    return f"<{NS}{mid}>"

# \ 예외 처리
def lit(s):
    return json.dumps(s.replace('\\"', '"'), ensure_ascii=False)

# 비교용. 문자 정규화(대소문자, 띄어쓰기)
def norm(s):
    return " ".join(s.split()).casefold()


# --------------------------------------------------------------------------- SPARQL client

# 로컬 Freebase에 SPARQL 쿼리 보내고 결과 캐싱
class Sparql:
    def __init__(self, endpoint, cache_path):
        self.endpoint = endpoint
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.mem = {}
        self.n_remote = 0
        self.n_rejected = 0
        self.rejected_log = open(os.path.join(os.path.dirname(cache_path), "rejected_queries.jsonl"), "a")
        self.slow_log = open(os.path.join(os.path.dirname(cache_path), "slow_queries.jsonl"), "a")

    def select(self, query, timeout_ms=None):
        key = hashlib.sha1(query.encode()).hexdigest()
        if key in self.mem:
            return self.mem[key]
        with self.lock:
            row = self.db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row is not None:
            res = json.loads(row[0])
        else:
            res = self._remote(query, timeout_ms)
            with self.lock:
                self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(res)))
                self.n_remote += 1
                if self.n_remote % 500 == 0:
                    self.db.commit()
        self.mem[key] = res
        return res

    def _remote(self, query, timeout_ms=None):
        params = {"query": query, "format": "application/sparql-results+json"}
        if timeout_ms:
            params["timeout"] = str(timeout_ms)  # Virtuoso aborts server-side and returns partial results
        data = urllib.parse.urlencode(params).encode()
        for attempt in range(5):
            try:
                t0 = time.time()
                with urllib.request.urlopen(self.endpoint, data, timeout=300) as r:
                    partial = r.headers.get("X-SQL-State") == "S1TAT"
                    bindings = json.load(r)["results"]["bindings"]
                if partial:
                    return None
                if time.time() - t0 > 2:
                    with self.lock:
                        self.slow_log.write(json.dumps({"sec": round(time.time() - t0, 1), "query": query[:600]}) + "\n")
                        self.slow_log.flush()
                return [{k: term(v) for k, v in b.items()} for b in bindings]
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if "Virtuoso" in body:  # query rejected (syntax / cost estimate): deterministic, do not retry
                    with self.lock:
                        self.n_rejected += 1
                        self.rejected_log.write(json.dumps({"error": body[:300], "query": query}) + "\n")
                    return []
                err = e
            except Exception as e:  # noqa: BLE001
                err = e
            if attempt == 4:
                raise RuntimeError(f"SPARQL failed: {err}\n{query}") from err
            time.sleep(2 ** attempt)

    def close(self):
        with self.lock:
            self.db.commit()
            self.db.close()
            self.rejected_log.close()
            self.slow_log.close()


# S,O kind 분류 
def term(v):
    """SPARQL JSON term -> compact node dict."""
    # rdf.freebase.com URI 형식에는 마지막에 MID(entity) or 'relation' 포함되므로 MID/'relation'만 추출 -> kind: entity
    # freebase 상에서 triple의 relation 자체도 KG의 entity가 될 수 있으므로 (P가 아닌) entity(S,O)로 분류
    # 그 외에 w3 URI 형식은 전체 URI 추출 -> kind: uri
    if v["type"] == "uri":
        val = v["value"]
        return {"kind": "entity", "id": val[len(NS):]} if val.startswith(NS) else {"kind": "uri", "id": val}
    # blank node 인데, 실제로 Freebase에서는 bnode 대신 CVT 노드(uri 형태)를 사용하므로 사용하지 않는 코드
    if v["type"] == "bnode":
        return {"kind": "bnode", "id": v["value"]}
    
    out = {"kind": "literal", "value": v["value"]}
    if "datatype" in v:
        out["datatype"] = v["datatype"]
    if "xml:lang" in v:
        out["lang"] = v["xml:lang"]
    return out

'''
{"kind":"entity", "id":"m.03_r3"}             → "m.03_r3"
{"kind":"uri", "id":"http://www.w3.org/..."}   → "http://www.w3.org/..."
{"kind":"literal", "value":"1961-08-04", ...}  → "lit:1961-08-04"
형태로 id와 value 만 추출해서 비교하기 위함
'''
def node_key(n):
    return n.get("id") or ("lit:" + n["value"])


# --------------------------------------------------------------------------- KG queries



class KG:
    def __init__(self, sparql):
        self.sp = sparql
    
    # MID : english name dictionary 생성
    def names(self, mids): 
        out = {}
        mids = sorted(set(mids))
        for i in range(0, len(mids), MAX_ANCHORS):
            vals = " ".join(mid_iri(m) for m in mids[i:i + MAX_ANCHORS])
            q = f"SELECT ?e ?n WHERE {{ VALUES ?e {{ {vals} }} ?e {NAME} ?n . FILTER(lang(?n)='en') }}"
            for b in self.sp.select(q):
                out.setdefault(b["e"]["id"], b["n"]["value"])
        return out
      
    # seed와 가까운 후보 정하는 코드
    # 후보가 여러개 남았을 때, 대표값 결정. 기준 -> sparql 부분에서 1hop 양방향, 2hop 같은 방향, 2hop 공통 이웃의 union 에 포함
    # 즉, seed와 2-hop 이내에 있는 entity를 mid로 결정
    def near_seeds(self, mids, seeds):
        """Subset of `mids` within 2 undirected hops of any seed (the methods' KG is the 2-hop neighbourhood)."""
        hit = set(mids) & set(seeds)
        mids = sorted(set(mids) - set(seeds))
        seeds = sorted(m for m in set(seeds) if not self.is_hub(m))
        if not mids or not seeds:
            return hit
        x = " ".join(mid_iri(m) for m in mids)
        sd = " ".join(mid_iri(m) for m in seeds)
        q = (f"SELECT DISTINCT ?x WHERE {{ VALUES ?x {{ {x} }} VALUES ?s {{ {sd} }} "
             "{ ?s ?p ?x } UNION { ?x ?p ?s } UNION { ?s ?p1 ?m . ?m ?p2 ?x } UNION { ?x ?p1 ?m . ?m ?p2 ?s } "
             "UNION { ?s ?p1 ?m . ?x ?p2 ?m } UNION { ?m ?p1 ?s . ?m ?p2 ?x } }")
        res = self.sp.select(q, timeout_ms=10000)
        return hit if res is None else hit | {b["x"]["id"] for b in res}
    
    # 들어오는 edge가 5000개 이상일 때, hub로 판별.
    def is_hub(self, mid, limit=5000):
        """Seeds such as gender 'Male' or a country make a 2-hop join explode; they carry no tie-break signal."""
        q = f"SELECT (COUNT(*) AS ?c) WHERE {{ SELECT ?s WHERE {{ ?s ?p {mid_iri(mid)} }} LIMIT {limit} }}"
        res = self.sp.select(q)
        return bool(res) and int(res[0]["c"]["value"]) >= limit
    
    # 나가는 edge 개수 = degree
    def degree(self, mid):
        q = f"SELECT (COUNT(*) AS ?c) WHERE {{ {mid_iri(mid)} ?p ?o }}"  # out-degree as a prominence proxy
        res = self.sp.select(q)
        return int(res[0]["c"]["value"]) if res else 0

    # 한쪽의 mids를 이미 찾았다는 가정 하에 src와 dst가 rel로 연결된 모든 edge를 찾는 함수
    def edges(self, src, rel, dst, directions=("fwd", "rev")):
        """Ground one hop. src/dst are specs:
             ("mids", [mid,...])  -- anchored set
             ("text", "string")   -- surface string from the method output
        Returns (list of (src_node, dst_node, direction), match_type).
        direction 'fwd' = KG triple (src, rel, dst); 'rev' = KG triple (dst, rel, src).
        """
        r = rel_iri(rel)
        combos = sorted(itertools.product(self._modes(src), self._modes(dst)), key=lambda c: RANK[c[0]] + RANK[c[1]])
        for ms, md in combos:
            if ms not in SELECTIVE and md not in SELECTIVE: # 예외 케이스
                continue  # would scan the whole relation
            if (ms == "name_ci" and md != "mids") or (md == "name_ci" and ms != "mids"): # 예외 케이스
                continue  # fuzzy matching only next to an anchored set
            pre_s, post_s = self._side("s", src[1], ms)
            pre_d, post_d = self._side("d", dst[1], md)
            res = []
            for d in directions:
                obj_mode = md if d == "fwd" else ms
                subj_mode = ms if d == "fwd" else md
                if subj_mode == "literal": # 예외 케이스
                    continue  # Freebase 특성상 literal 에서 나오는 edge 없으므로, literals are never subjects (and Virtuoso rejects the plan)
                if obj_mode not in SELECTIVE and subj_mode not in SELECTIVE: # 예외 케이스
                    continue
                edge = f"?s {r} ?d ." if d == "fwd" else f"?d {r} ?s ." # 양방향
                q = "SELECT DISTINCT ?s ?d WHERE { " + " ".join(pre_s + pre_d) + f" {edge} " + \
                    " ".join(post_s + post_d) + " } LIMIT 1000"
                rows = self.sp.select(q, timeout_ms=20000)
                for b in rows or []:  # None = server timeout (only seen for fuzzy matching next to hubs)
                    res.append((b["s"], b["d"], d))
            if res:
                return res, "/".join(m for m in (ms, md) if m != "mids")
        return [], None

    @staticmethod
    def _modes(spec):
        kind, val = spec
        if kind == "mids":
            return ["mids"]
        if MID_RE.match(val):
            return ["mid"] # CVT node (kind는 mid가 아니지만, value가 mid 형태)
        return ["name", "literal", "alias", "name_ci"]

    @staticmethod
    def _side(var, val, mode):
        """-> (patterns placed before the edge, patterns placed after it)."""
        if mode == "mids":
            return [f"VALUES ?{var} {{ {' '.join(mid_iri(m) for m in val)} }}"], []
        if mode == "mid":
            return [f"VALUES ?{var} {{ {mid_iri(val)} }}"], []
        if mode == "name":
            return [f"?{var} {NAME} {lit(val)}@en ."], []
        if mode == "alias":
            return [f"?{var} {ALIAS} {lit(val)}@en ."], []
        if mode == "literal": # Freebase의 literal 값 자체가 찾은 val(surface string) 과 같은 경우
            return [], [f"FILTER(isLiteral(?{var}) && str(?{var}) = {lit(val)})"]
        if mode == "name_ci":  # any language, case-insensitive
            return [], [f"?{var} {NAME} ?n_{var} . FILTER(lcase(str(?n_{var})) = {lit(val.lower())})"]
        raise ValueError(mode)


# --------------------------------------------------------------------------- parsing

# RoG, GNN-RAG reasoning path : e1->r1->e2 형태
def parse_paths(text):
    m = re.search(r"Reasoning Paths:\n(.*?)\n\nQuestion:", text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).split("\n"): # reasoning paths만 추출
        line = line.strip()
        if not line:
            continue
        toks = line.split(" -> ")
        ok = len(toks) % 2 == 1 and len(toks) >= 3 and all(REL_RE.match(t) for t in toks[1::2]) # relation 형식 검사
        out.append({"raw": line, "nodes": toks[0::2] if ok else None, "rels": toks[1::2] if ok else None})
    return out

# SubgraphRAG, ParallaxRAG reasoning path (h,r,t) 형태
def parse_triplets(text):
    m = re.search(r"Triplets:\n(.*?)(?:\n\n|$)", text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).split("\n"): # triplets만 추출
        line = line.strip()
        if not (line.startswith("(") and line.endswith(")")): # 예외 처리 '(' 시작 ')' 끝 아닌 경우
            continue
        body = line[1:-1]
        mm = TRIPLE_RE.search(body) # , (relation) , 형태
        if mm:
            out.append({"raw": line, "h": body[:mm.start()], "r": mm.group(1), "t": body[mm.end():]})
        else:
            out.append({"raw": line, "h": None, "r": None, "t": None})
    return out

# 모델별로 분기
def load_items(method, rec):
    if method in ("rog", "gnnrag"):
        return "path", parse_paths(rec["input"])
    if method == "subgraphrag":
        return "triple", parse_triplets(rec["user_query"])
    if method == "parallaxrag":
        return "triple", parse_triplets(rec["input"])
    raise ValueError(method)

# seed MID 설정
def load_seeds(gold_path):
    """question id -> set of MID constants appearing in the dataset SPARQL (topic / constraint entities)."""
    seeds = {}
    with open(gold_path) as f:
        for line in f:
            r = json.loads(line)
            qid = re.sub(r"\.P\d+$", "", r["id"]) # .P 로 나눠져있는 sparql 통합
            seeds.setdefault(qid, set()).update(re.findall(r"ns:([mg]\.[0-9a-z_]+)", r["sparql"])) # 해당되는 모든 관련 MID seed에 추가
    return seeds 
    # dataset에 있는 topic entity 만 들어가는 게 아니라 제약 entity도 다 포함 (SPARQL 기준)


# --------------------------------------------------------------------------- resolution

# node에 "surface", "name" 키 추가
def out_node(n, surface, names): # surface 는 reasoning path 에서 실제로 찾은 문자열
    d = dict(n)
    d["surface"] = surface
    if d["kind"] == "entity":
        d["name"] = names.get(d["id"])
    return d
# ex) reasoning path에서 "hanolulu" 찾은 경우, name_ci로 m.03gh4에 매칭
# {"kind":"entity", "id":"m.03gh4", "surface":"honolulu", "name":"Honolulu"}

class Resolver:
    def __init__(self, kg):
        self.kg = kg

    # ---- shared helpers
    # (src, dst, dir) 형태 edge -> KG 방향 기준 triple dictionary 로 변환
    def _canonical(self, s, d, direction, rel, s_surf, d_surf, names):
        if direction == "fwd":
            return {"subject": out_node(s, s_surf, names), "predicate": rel, "object": out_node(d, d_surf, names),
                    "direction": "forward"}
        return {"subject": out_node(d, d_surf, names), "predicate": rel, "object": out_node(s, s_surf, names),
                "direction": "reverse"}
    # ex) reasoning path 출력: James K.Polk -> office_holder -> m.04j60kc
    # 실제 Freebase에 저장된 트리플: (m04j60kc, office_holder, m.042f1)
    # edges() 결과: s=m.042f1, d=m.04j60kc, dir="rev" => reasoning path 기준 node 저장
    # _canonical 결과:
    # {"subject": {"id":"m.04j60kc", "surface":"m.04j60kc", ...},
    #   "predicate": "government.government_position_held.office_holder",
    #   "object":  {"id":"m.042f1", "surface":"James K. Polk", "name":"James K. Polk"},
    #    direction": "reverse"}
    # => KG 기준 SPO dictionary 형태 저장


    # (src, dst, dir) 형태 edge -> KG 방향 기준 (subject key, object key) tuple 로 변환
    @staticmethod
    def _kg_pair(edge):
        s, d, dr = edge
        return (node_key(s), node_key(d)) if dr == "fwd" else (node_key(d), node_key(s))

    # 타입도 같고 연결 구조가 동일해서 path 유효성 검사에서 안 걸러진 후보(ambiguous) 중 대표값 결정
    def _pick(self, cands, prefer, seeds):
        """Choose a primary among equally valid candidates (all of them are still reported).
        cands: list of candidate tuples of node keys; prefer: keys grounded elsewhere in the same sample."""
        if len(cands) == 1:
            return 0, "unique"
        top = list(range(len(cands)))

        # 기준 1 - sample_consistency(같은 질문의 다른 path나 triple에서 이미 확정된 MID인지?)
        for how, score in (("sample_consistency", lambda c: sum(k in prefer for k in c)),):
            scores = {i: score(cands[i]) for i in top}
            best = max(scores.values())
            top = [i for i in top if scores[i] == best]
            if len(top) == 1:
                return top[0], how
        # 기준 2 - degree
        deg = {i: sum(self.kg.degree(k) for k in cands[i] if MID_RE.match(k)) for i in top}
        return max(top, key=lambda i: deg[i]), "degree_tiebreak"

    # ---- paths (RoG / GNN-RAG)
    def resolve_path(self, item, seeds, seed_names):
        nodes, rels = item["nodes"], item["rels"]
        if nodes is None: # 파싱 처리 실패
            return {"raw": item["raw"], "status": "parse_error", "triples": []}
        head = nodes[0] # 시작 노드
        start = [m for m in seeds if norm(seed_names.get(m, "")) == norm(head)] or ([head] if MID_RE.match(head) else [])
        
        # start_mode : 
        # 1) seed: seed에 있는 mid와 첫 노드 이름 같을때
        # 2) seed_homonym: seed로 시작했는데 첫 hop부터 edge 못 찾아서 이름만 같은 다른 엔티티로 시작했을 때
        # 3) mid : 첫 노드 문자열 자체가 m.xxx 형태
        # 4) name_lookup : 나머지 경우
        start_mode = "seed" if start and start[0] in seeds else "mid" if start else "name_lookup"
        if not start:
            start = self._homonym_starts(head, rels[0], nodes[1])
        layers, hop_edges, broken_at = self._walk(start, nodes, rels)
        if broken_at == 0 and start_mode == "seed":
            # The head only shares its name with the seed (e.g. the book "Thomas Jefferson"): the method's
            # name-keyed graph merged the two. Re-ground the whole chain from the homonyms instead.
            alt = [m for m in self._homonym_starts(head, rels[0], nodes[1]) if m not in seeds]
            if alt:
                start_mode = "seed_homonym"
                layers, hop_edges, broken_at = self._walk(alt, nodes, rels)

        # 역방향 pruning : 중간에 path 끊긴 경우에는 제외
        alive_hops = []
        alive_next = layers[len(hop_edges)]
        for found, mtype in reversed(hop_edges):
            kept = [e for e in found if node_key(e[1]) in alive_next]
            alive_hops.append((kept, mtype))
            alive_next = {node_key(e[0]) for e in kept}
        alive_hops.reverse()

        # enumerate chains (bounded)
        chains = [[]]
        for kept, _ in alive_hops:
            nxt = []
            for c in chains:
                for e in kept:
                    if not c or node_key(c[-1][1]) == node_key(e[0]): # 앞 엣지 도착 = 현재 엣지 출발
                        nxt.append(c + [e])
            chains = nxt[:MAX_CHAINS]
        mids = {node_key(n) for c in chains for e in c for n in (e[0], e[1]) if n["kind"] == "entity"}
        names = self.kg.names(mids) if mids else {}
        prefer = set(seeds)
        idx, how = (0, "unique") if len(chains) <= 1 else self._pick(
            [tuple(node_key(e[1]) for e in c) for c in chains], prefer, seeds)
        primary = chains[idx] if chains and chains[0] else []

        # hop 단위 ambiguous 판별
        triples = []
        for i, e in enumerate(primary):
            t = self._canonical(e[0], e[1], e[2], rels[i], nodes[i], nodes[i + 1], names)
            t["match"] = alive_hops[i][1]
            alts = sorted({self._kg_pair(x) for x in alive_hops[i][0]})
            t["status"] = "resolved" if len(alts) == 1 else "ambiguous"
            if len(alts) > 1:
                t["n_candidates"] = len(alts)
                t["candidates"] = [list(c) for c in alts[:20]]
            triples.append(t)
            
        # hops after a break: ground independently (no chain constraint)
        for i in range(len(primary), len(rels)):
            triples.append(self._resolve_free(nodes[i], rels[i], nodes[i + 1], ("fwd", "rev"), prefer, seeds, mode="path_broken"))

        if broken_at is not None or not primary:
            status = "partial" if any(t["status"] in ("resolved", "ambiguous") for t in triples) else "unresolved"
        else:
            status = "resolved" if len(chains) == 1 else "ambiguous"
        out = {"raw": item["raw"], "status": status, "start": start_mode, "triples": triples,
               "node_ids": [primary[0][0].get("id")] + [node_key(e[1]) for e in primary] if primary else None}
        if len(chains) > 1:
            out["n_chains"] = len(chains)
            out["pick"] = how
        return out

    def _homonym_starts(self, head, rel, nxt):
        res, _ = self.kg.edges(("text", head), rel, ("text", nxt))
        return sorted({s["id"] for s, _, _ in res if s["kind"] == "entity"})

    def _walk(self, start, nodes, rels):
        """Forward beam: layer i+1 = nodes named nodes[i+1] linked by rels[i] to a node of layer i."""
        layers, hop_edges = [set(start)], []
        for i, rel in enumerate(rels):
            src = sorted(k for k in layers[-1] if MID_RE.match(k))
            found, mtype = [], None
            for j in range(0, len(src), MAX_ANCHORS):
                f, mt = self.kg.edges(("mids", src[j:j + MAX_ANCHORS]), rel, ("text", nodes[i + 1]))
                found += f
                mtype = mtype or mt
            if not found:
                return layers, hop_edges, i
            hop_edges.append((found, mtype))
            layers.append({node_key(d) for _, d, _ in found})
        return layers, hop_edges, None

    # ---- unanchored single triple
    def _resolve_free(self, h, r, t, directions, prefer, seeds, mode):
        res, mtype = self.kg.edges(("text", h), r, ("text", t), directions)
        if not res:
            return {"subject": {"surface": h}, "predicate": r, "object": {"surface": t}, "status": "unresolved",
                    "resolution": mode}
        out = self._finish(res, mtype, mode, r, h, t, prefer, seeds)
        return out

    # ---- triple sets (SubgraphRAG / ParallaxRAG)
    def resolve_triples(self, items, seeds, seed_names):
        known = {}  # norm(surface) -> set of node keys
        for m in seeds:
            if m in seed_names:
                known.setdefault(norm(seed_names[m]), set()).add(m)
        results = [None] * len(items)
        pending = []
        for i, it in enumerate(items):
            if it["h"] is None:
                results[i] = {"raw": it["raw"], "status": "parse_error"}
            else:
                pending.append(i)

        def learn(surface, key):
            if MID_RE.match(key):
                known.setdefault(norm(surface), set()).add(key)

        def anchored(it):
            for side in ("h", "t"):
                s = it[side]
                anchor = sorted(known.get(norm(s), set())) or ([s] if MID_RE.match(s) else [])
                if not anchor:
                    continue
                other = it["t"] if side == "h" else it["h"]
                spec_a = ("mids", anchor[:MAX_ANCHORS])
                if side == "h":
                    res, mt = self.kg.edges(spec_a, it["r"], ("text", other), ("fwd",))
                    if not res:
                        res, mt = self.kg.edges(spec_a, it["r"], ("text", other), ("rev",))
                else:
                    # anchor is the tail: query with tail fixed, returned as (h, t, fwd)
                    res, mt = self.kg.edges(("text", other), it["r"], spec_a, ("fwd",))
                    if not res:
                        res, mt = self.kg.edges(("text", other), it["r"], spec_a, ("rev",))
                if res:
                    return res, mt, side
            return None

        # 1) propagate from seeds through the retrieved subgraph
        progress = True
        while progress and pending:
            progress = False
            still = []
            for i in pending:
                it = items[i]
                has_anchor = (norm(it["h"]) in known or MID_RE.match(it["h"]) or
                              norm(it["t"]) in known or MID_RE.match(it["t"]))
                got = anchored(it) if has_anchor else None
                if not got:
                    still.append(i)
                    continue
                res, mt, side = got
                results[i] = self._finish_triple(it, res, mt, f"anchored_{'head' if side == 'h' else 'tail'}", known, seeds)
                progress = True
                tr = results[i]
                for role, surf in (("subject", "surface"), ("object", "surface")):
                    n = tr[role]
                    if n.get("kind") == "entity":
                        learn(n[surf], n["id"])
            pending = still

        # 2) unanchored: relation-constrained grounding on both names
        prefer = {k for v in known.values() for k in v}
        for i in pending:
            it = items[i]
            tr = self._resolve_free(it["h"], it["r"], it["t"], ("fwd", "rev"), prefer, seeds, mode="relation_constrained")
            tr["raw"] = it["raw"]
            results[i] = tr
        return results

    def _finish_triple(self, it, res, mtype, mode, known, seeds):
        prefer = {k for v in known.values() for k in v}
        out = self._finish(res, mtype, mode, it["r"], it["h"], it["t"], prefer, seeds)
        out["raw"] = it["raw"]
        return out

    def _finish(self, res, mtype, mode, rel, h_surf, t_surf, prefer, seeds):
        """res: edges (src, dst, dir) with src matching h_surf. Dedupe on the KG (subject, object) pair."""
        cands = {}
        for e in res:
            cands.setdefault(self._kg_pair(e), e)
        keys = sorted(cands)
        idx, how = self._pick(keys, prefer, seeds)
        s, d, dr = cands[keys[idx]]
        names = self.kg.names({k for c in keys for k in c if MID_RE.match(k)})
        out = self._canonical(s, d, dr, rel, h_surf, t_surf, names)
        out.update({"match": mtype, "resolution": mode, "status": "resolved" if len(keys) == 1 else "ambiguous"})
        if len(keys) > 1:
            out["n_candidates"] = len(keys)
            out["pick"] = how
            out["candidates"] = [list(c) for c in keys[:20]]
        return out


# --------------------------------------------------------------------------- driver


def process_record(resolver, method, rec, seeds_all):
    kind, items = load_items(method, rec)
    seeds = sorted(seeds_all.get(rec["id"], set()))
    seed_names = resolver.kg.names(seeds) if seeds else {}
    if kind == "path":
        outs = [resolver.resolve_path(it, seeds, seed_names) for it in items]
    else:
        outs = resolver.resolve_triples(items, seeds, seed_names)
    status = {}
    for o in outs:
        status[o["status"]] = status.get(o["status"], 0) + 1
    return {
        "id": rec["id"],
        "question": rec["question"],
        "method": method,
        "item_type": kind,
        "seed_entities": [{"id": m, "name": seed_names.get(m)} for m in seeds],
        "num_items": len(outs),
        "status_counts": status,
        "items": outs,
    }


def run(method, dataset, args, sparql):
    in_path = os.path.join(args.data_dir, f"{method}_paths", f"{dataset}.{method}.jsonl")
    gold_path = os.path.join(args.data_dir, "gold_paths", f"{dataset}.gold.jsonl")
    out_dir = os.path.join(args.out_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}.{method}.mid.jsonl")

    seeds_all = load_seeds(gold_path)
    with open(in_path) as f:
        recs = [json.loads(line) for line in f]
    if args.limit:
        recs = recs[:args.limit]
    resolver = Resolver(KG(sparql))
    with ThreadPoolExecutor(args.workers) as ex, open(out_path + ".tmp", "w") as fout:
        totals = {}
        for out in tqdm(ex.map(lambda r: process_record(resolver, method, r, seeds_all), recs),
                        total=len(recs), desc=f"{dataset}/{method}"):
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            for k, v in out["status_counts"].items():
                totals[k] = totals.get(k, 0) + v
    os.replace(out_path + ".tmp", out_path)
    print(f"[{dataset}/{method}] -> {out_path}  {totals}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="all", choices=METHODS + ["all"])
    p.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    p.add_argument("--data_dir", default=os.path.join(root, "data"))
    p.add_argument("--out_dir", default=os.path.join(root, "data", "mid_paths"))
    p.add_argument("--endpoint", default="http://localhost:3001/sparql")
    p.add_argument("--cache", default=os.path.join(root, "data", "mid_paths", ".cache", "sparql_cache.sqlite"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    sparql = Sparql(args.endpoint, args.cache)
    try:
        for ds in DATASETS if args.dataset == "all" else [args.dataset]:
            for m in METHODS if args.method == "all" else [args.method]:
                run(m, ds, args, sparql)
    finally:
        sparql.close()


if __name__ == "__main__":
    main()
