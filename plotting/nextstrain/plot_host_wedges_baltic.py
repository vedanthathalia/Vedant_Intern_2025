import argparse
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import baltic as bt

DEFAULT_TREE = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/tree_named.nwk"
DEFAULT_META = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/metadata_with_mutations.tsv"
DEFAULT_OUT  = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/host_wedges.png"
DEFAULT_ID   = "strain"
DEFAULT_HOST = "host"
DEFAULT_MIN  = 10
DEFAULT_WIDTH_SCALE = 0.5
DEFAULT_DPI  = 200
DEFAULT_TITLE = "Phylogeny by Host"

HOST_COLORS = {
    "human":   "#3b82f6",
    "avian":   "#f6c945",
    "other":   "#a855f7",
    "unknown": "#9ca3af",
}

def guess_sep(p: Path) -> str:
    return "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","

def load_tree(tree_path: str):
    p = Path(tree_path)
    if p.suffix.lower() in {".nwk", ".newick", ".tree", ".tre"}:
        ll = bt.loadNewick(str(p), absoluteTime=False)
    else:
        ll = bt.loadNexus(str(p), absoluteTime=False)
    ll.treeStats()
    return ll

def build_host_lookup(meta_path: str, id_col: str, host_col: str, strip_version: bool = True):
    df = pd.read_csv(meta_path, sep=guess_sep(Path(meta_path)))
    if id_col not in df.columns: raise SystemExit(f"'{id_col}' missing from metadata columns.")
    if host_col not in df.columns: raise SystemExit(f"'{host_col}' missing from metadata columns.")
    ids = df[id_col].astype(str)
    if strip_version:
        ids = ids.str.replace(r"\.\d+$", "", regex=True)
    hosts = (
        df[host_col].astype(str).str.lower().fillna("unknown")
        .replace({"humans":"human","bird":"avian","na":"unknown","nan":"unknown","?":"unknown"})
    )
    return dict(zip(ids, hosts))

def attach_host_to_tips_and_make_map(ll, host_lookup, strip_version=True):
    tip_host_map = {}
    n_tips = matched = 0
    for leaf in ll.getExternal():
        if not leaf.is_leaf(): continue
        n_tips += 1
        name = str(leaf.name)
        key = name.split(".")[0] if strip_version else name
        host = host_lookup.get(key, "unknown")
        leaf.traits["host"] = host
        tip_host_map[name] = host
        if host != "unknown": matched += 1
    return n_tips, matched, tip_host_map

def mono_host_info(node, tip_host_map):
    names = getattr(node, "leaves", None)
    if names is None: return (False, "", 0)
    size = len(names)
    hosts = set()
    for nm in names:
        h = tip_host_map.get(nm, "unknown")
        if h != "unknown": hosts.add(h)
        if len(hosts) > 1: return (False, "", size)
    if len(hosts) == 1:
        return (True, next(iter(hosts)), size)
    return (False, "", size)

def main():
    ap = argparse.ArgumentParser(description="Plot triangles for single-host clades (baltic).", add_help=True)
    ap.add_argument("--tree", default=DEFAULT_TREE)
    ap.add_argument("--metadata", default=DEFAULT_META)
    ap.add_argument("--outfile", default=DEFAULT_OUT)
    ap.add_argument("--id-column", default=DEFAULT_ID)
    ap.add_argument("--host-column", default=DEFAULT_HOST)
    ap.add_argument("--min-size", type=int, default=DEFAULT_MIN)
    ap.add_argument("--width-scale", type=float, default=DEFAULT_WIDTH_SCALE)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--title", default=DEFAULT_TITLE)
    args = ap.parse_args([])

    ll = load_tree(args.tree)
    host_lookup = build_host_lookup(args.metadata, args.id_column, args.host_column, strip_version=True)
    n_tips, matched, tip_host_map = attach_host_to_tips_and_make_map(ll, host_lookup, strip_version=True)
    print(f"Tip host assignment: matched {matched} / {n_tips} tips")

    def colour(obj):
        return HOST_COLORS.get(obj.traits.get("host","unknown"), HOST_COLORS["unknown"])

    ll.uncollapseSubtree()

    collapsed_count = 0
    clade_host = {}
    min_size = args.min_size

    while True:
        candidates = []
        for node in ll.getBranches(lambda b: b.is_node() and b.parent is not None):
            ok, host, size = mono_host_info(node, tip_host_map)
            if ok and size >= min_size:
                candidates.append((size, host, node))
        if not candidates: break
        candidates.sort(key=lambda x: x[0], reverse=True)
        size, host, node = candidates[0]
        if node not in ll.Objects: continue
        try:
            cname = f"{host}_clade_{collapsed_count+1}_n={size}"
            node.traits["host"] = host
            ll.collapseSubtree(node, cname, widthFunction=lambda k: len(k.leaves) * args.width_scale)
            clade_host[cname] = host
            collapsed_count += 1
        except Exception as e:
            print(f"skip node (size={size}) due to: {e}")
            continue

    print(f"Collapsed {collapsed_count} single-host clades (min size = {min_size}).")

    height = max(10, min(40, int(n_tips/200)))
    fig = plt.figure(figsize=(6, height), facecolor="w")
    gs = gridspec.GridSpec(1, 1, wspace=0.0)
    ax = plt.subplot(gs[0], facecolor="w")

    ll.plotTree(ax, colour=colour, zorder=10)
    ll.plotPoints(ax, target=lambda k: k.is_leaf(), size=12, colour=colour, zorder=14)

    def is_clade(obj): return isinstance(obj, bt.clade)

    total_objs = len(ll.Objects)
    for k in ll.getExternal(is_clade):
        start_x, end_x, y = k.x, k.lastHeight, k.y
        host = clade_host.get(k.name, k.traits.get("host","unknown"))
        c = HOST_COLORS.get(host, HOST_COLORS["unknown"])
        half_thick = 0.0005 * total_objs
        poly = Polygon(
            ((start_x, y-half_thick),
             (start_x, y+half_thick),
             (end_x,   y+k.width/2.0),
             (end_x,   y-k.width/2.0)),
            facecolor=c, edgecolor="k", linewidth=0.6, zorder=12
        )
        ax.add_patch(poly)

    ax.set_title(args.title, fontsize=14)
    ax.set_ylim(-2, ll.ySpan+2)
    for loc in ax.spines: ax.spines[loc].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    present_hosts = set()
    for leaf in ll.getExternal():
        if leaf.is_leaf():
            present_hosts.add(leaf.traits.get("host","unknown"))
    for h in list(clade_host.values()):
        present_hosts.add(h)
    legend_hosts = [h for h in present_hosts if h not in {"unknown","other"}]
    legend_handles = [mpl.patches.Patch(facecolor=HOST_COLORS[h], edgecolor="k", label=h) for h in sorted(legend_hosts)]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Host", loc="upper right", frameon=False)

    out = Path(args.outfile); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=args.dpi)
    print(f"Saved figure {out.resolve()}")

if __name__ == "__main__":
    main()
