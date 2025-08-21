import csv
from collections import defaultdict
import re
import matplotlib.pyplot as plt

INPUT_CSV = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/closest_sequence/nearest_avian_by_subtype.csv"
OUTPUT_PNG = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/Figures/avian_distance_violinplot.png"

def subtype_sort_key(subtype):
    match = re.match(r"H(\d+)", subtype.upper())
    if match:
        return (0, int(match.group(1)))
    else:
        return (1, subtype)

def main():
    by_subtype = defaultdict(list)
    with open(INPUT_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            st = r["subtype"].strip()
            try:
                d = float(r["avian_distance"])
            except ValueError:
                continue
            by_subtype[st].append(d)

    subtypes = sorted((k for k, v in by_subtype.items() if v), key=subtype_sort_key)
    if not subtypes:
        print("No data")
        return

    data = [by_subtype[s] for s in subtypes]

    plt.figure(figsize=(max(6, 0.7*len(subtypes)), 6))
    parts = plt.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    plt.xticks(range(1, len(subtypes)+1), subtypes, rotation=45, ha='right')
    plt.ylabel("Distance (human to nearest avian)")
    plt.xlabel("Subtype")
    plt.title("Human Nearest Avian Distances by Subtype")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()
    print(f"Saved figure {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
