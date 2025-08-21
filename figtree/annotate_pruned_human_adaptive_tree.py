from Bio import Phylo
import pandas as pd

full_tree_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/all_pb2_aligned.newick"
subset_ids_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/Tree/interesting_ids.txt"
mutations_csv = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/Compared_Mutations_Output/compared_mutations_output.csv"
output_tree_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/Tree/annotated_pruned_tree.nex"

tree = Phylo.read(full_tree_path, "newick")
with open(subset_ids_path) as f:
    target_ids = set(line.strip() for line in f if line.strip())

# Prune tree
all_terminals = {term.name for term in tree.get_terminals()}
to_remove = all_terminals - target_ids
for name in to_remove:
    tree.prune(target=name)

Phylo.write(tree, output_tree_path, "nexus")

# Map Mutations
mut_df = pd.read_csv(mutations_csv)
mut_map = dict(zip(mut_df["Sequence_ID"], mut_df["Matched_FluMut_Mutations"].fillna("N/A").astype(str)))

with open(output_tree_path, "a") as f:
    f.write("\nBEGIN TRAITS;\n")
    f.write("    Dimensions NTax={} NChar=1;\n".format(len(tree.get_terminals())))
    f.write("    Format labels=yes missing=? separator=Comma;\n")
    f.write("    Matrix\n")
    for clade in tree.get_terminals():
        accession = clade.name
        mutation = mut_map.get(accession, "N/A").replace(" ", "").replace(";", "|") or "N/A"
        f.write(f"    {accession} {mutation}\n")
    f.write("    ;\nEND;\n")
