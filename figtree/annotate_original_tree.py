import baltic as bt
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os

tree_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/Tree/pb2_tree_flu_model.treefile"
csv_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_human_model/global_test_log_likelihoods.csv"
output_nexus = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/Tree/annotated_pb2_tree_flu_model.nex"
unaligned_fasta_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/NEW_HA_Subtype_Unaligned_FASTAs"

df = pd.read_csv(csv_path)
df = df.set_index("accession")

metadata = {}
for acc, row in df.iterrows():
    if row["label_human"] == 1 and row["label_avian"] == 1:
        host = "both"
    elif row["label_human"] == 1:
        host = "human"
    elif row["label_avian"] == 1:
        host = "avian"
    elif row["label_other"] == 1:
        host = "other"
    else:
        host = "unknown"

    metadata[acc] = {
        "log_likelihood": round(row["log_likelihood"], 4),
        "host": host
    }

tree = bt.loadNewick(tree_path)

for leaf in tree.Objects:
    if leaf.branchType == "leaf":
        name = leaf.name
        if name in metadata:
            leaf.traits["log_likelihood"] = metadata[name]["log_likelihood"]
            leaf.traits["host"] = metadata[name]["host"]
        else:
            print(f"{name} not found in metadata")

def build_annotated_newick(node):
    if node.branchType == "leaf":
        traits = node.traits
        annotations = ",".join(f'{k}="{v}"' for k, v in traits.items())
        return f'{node.name}[&{annotations}]:{round(node.length, 6)}'
    else:
        children = [build_annotated_newick(child) for child in node.children]
        return f'({",".join(children)}):{round(node.length, 6)}'

annotated_tree = build_annotated_newick(tree.root) + ';'

with open(output_nexus, "w") as f:
    f.write("#NEXUS\n\n")
    f.write("Begin trees;\n")
    f.write(f"   tree TREE1 = [&R] {annotated_tree}\n")
    f.write("End;\n")

print(f"Annotated NEXUS tree written to: {output_nexus}")