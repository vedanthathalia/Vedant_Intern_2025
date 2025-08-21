import csv
from collections import defaultdict
import baltic as bt

TREE_PATH = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/tree_named.nwk"
METADATA_TSV = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/metadata_with_mutations.tsv"
OUTPUT_CSV = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/closest_sequence/nearest_avian_by_subtype.csv"
SUBTYPES = None
# SUBTYPES = {"H1", "H3"}

def build_tip_map(tree):
    return {getattr(n, 'name', None): n for n in tree.Objects if n.branchType == 'leaf'}

def path_to_root(node):
    out, dist, cur = [], 0.0, node
    while cur is not None:
        out.append((cur, dist))
        dist = dist + getattr(cur, 'length', 0.0)
        cur = getattr(cur, 'parent', None)
    return out

def distance(a, b):
    pa = path_to_root(a); pb = path_to_root(b)
    dist_a = {id(n): d for n, d in pa}
    ancestors_a = {id(n) for n, _ in pa}
    for n, dist_b in pb:
        if id(n) in ancestors_a:
            return dist_a[id(n)] + dist_b
    return dist_a[id(pa[-1][0])] + pb[-1][1]

def descendants_in_subtype(node, subtype, meta, tip_nodes):
    for leaf_name in getattr(node, "leaves", []):
        t = tip_nodes.get(leaf_name)
        if not t: continue
        m = meta.get(leaf_name)
        if not m: continue
        if m.get('subtype') == subtype:
            yield t

def first_ancestor_with_avian(hnode, subtype, meta, tip_nodes):
    cur = hnode
    while cur is not None:
        for t in descendants_in_subtype(cur, subtype, meta, tip_nodes):
            if meta.get(getattr(t, 'name', ''), {}).get('host') == 'avian':
                return cur
        cur = getattr(cur, 'parent', None)
    return None

def main():
    meta = {}
    with open(METADATA_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            sid = r.get("strain")
            if not sid: continue
            meta[sid] = {
                "host": (r.get("host") or "").strip().lower(),
                "subtype": (r.get("subtype") or "").strip()
            }

    T = bt.loadNewick(TREE_PATH, absoluteTime=False)
    tip_nodes = build_tip_map(T)

    all_st = {v["subtype"] for v in meta.values() if v.get("subtype")}
    target_st = set(SUBTYPES) if SUBTYPES else all_st

    # Tips by subtype and host
    by_subtype = defaultdict(lambda: {"human": [], "avian": []})
    for sid, m in meta.items():
        if sid not in tip_nodes: continue
        st = m.get("subtype")
        if st not in target_st: continue
        if m.get("host") == "human": by_subtype[st]["human"].append(sid)
        elif m.get("host") == "avian": by_subtype[st]["avian"].append(sid)

    rows = []
    for st, groups in by_subtype.items():
        avian_nodes = [tip_nodes[nm] for nm in groups["avian"] if nm in tip_nodes]
        if not avian_nodes:
            continue

        for hname in groups["human"]:
            hnode = tip_nodes.get(hname)
            if not hnode: continue

            anchor = first_ancestor_with_avian(hnode, st, meta, tip_nodes)
            if anchor is None:
                continue

            candidates = [t for t in descendants_in_subtype(anchor, st, meta, tip_nodes)
                          if meta.get(t.name, {}).get('host') == 'avian']
            if not candidates:
                continue

            best_tip, best_dist = None, float("inf")
            for a in candidates:
                d = distance(hnode, a)
                if d < best_dist:
                    best_dist, best_tip = d, a

            rows.append({
                "subtype": st,
                "human_strain": hname,
                "closest_avian": getattr(best_tip, "name", ""),
                "avian_distance": f"{best_dist:.10f}",
                "anchor_node": getattr(anchor, "name", getattr(anchor, "id", "")),
                "anchor_tips": str(len(getattr(anchor, "leaves", [])))
            })

    fieldnames = ["subtype","human_strain","closest_avian","avian_distance","anchor_node","anchor_tips"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} rows {OUTPUT_CSV}")

if __name__ == "__main__":
    main()