import csv
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

INPUT_CSV = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/closest_sequence/nearest_avian_by_subtype.csv"
OUTPUT_PNG = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/Figures/avian_distance_violinplots.png"

def subtype_sort_key(subtype):
    match = re.match(r"H(\d+)", subtype.upper())
    if match:
        return (0, int(match.group(1)))
    else:
        return (1, subtype)

def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["avian_distance", "subtype"])
    df["avian_distance"] = pd.to_numeric(df["avian_distance"], errors="coerce")
    df = df.dropna(subset=["avian_distance"])
    
    subtypes_sorted = sorted(df["subtype"].unique(), key=subtype_sort_key)
    df["subtype"] = pd.Categorical(df["subtype"], categories=subtypes_sorted, ordered=True)
    
    plt.figure(figsize=(max(6, 0.7*len(subtypes_sorted)), 6))
    sns.violinplot(
        x="subtype",
        y="avian_distance",
        data=df,
        order=subtypes_sorted,
        inner="box",
        scale="width",
        color="steelblue"
    )
    
    plt.ylabel("Distance (human to nearest avian)")
    plt.xlabel("Subtype")
    plt.title("Human Nearest Avian Distances by Subtype")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()
    print(f"Saved figure {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
