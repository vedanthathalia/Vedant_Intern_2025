import os
import pandas as pd
from collections import defaultdict
from Bio import SeqIO

aligned_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/NEW_HA_Subtype_Aligned_FASTAs"
output_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/NEW_Compared_Mutations_Output"
combined_output_path = os.path.join(output_dir, "all_mutations.csv")

os.makedirs(output_dir, exist_ok=True)

all_mutation_rows = []

for file in os.listdir(aligned_dir):
    if not file.endswith(".fasta"):
        continue

    subtype = file.replace(".fasta", "")
    fasta_path = os.path.join(aligned_dir, file)
    records = list(SeqIO.parse(fasta_path, "fasta"))

    if len(records) < 2:
        print(f"Skipping {subtype}, not enough sequences.")
        continue

    ref_record = records[0]
    ref_seq = str(ref_record.seq)

    mutations_per_seq = defaultdict(list)

    for record in records[1:]:
        query_seq = str(record.seq)
        for i, (ref_aa, query_aa) in enumerate(zip(ref_seq, query_seq)):
            if ref_aa != query_aa and ref_aa != '-' and query_aa != '-':
                mutation = f"{ref_aa}{i+1}{query_aa}"
                mutations_per_seq[record.id].append(mutation)

    subtype_output_path = os.path.join(output_dir, f"{subtype}_mutations.csv")
    with open(subtype_output_path, "w") as f:
        f.write("accession_id,mutations\n")
        for record in records[1:]:
            muts = mutations_per_seq.get(record.id, [])
            mut_str = '; '.join(muts) if muts else 'N/A'
            f.write(f"{record.id},{mut_str}\n")
            all_mutation_rows.append((record.id, mut_str))

    print(f"Saved mutations for {subtype} → {subtype_output_path}")

with open(combined_output_path, "w") as f:
    f.write("accession_id,mutations\n")
    for acc, muts in all_mutation_rows:
        f.write(f"{acc},{muts}\n")

print(f"\nAll mutations written to: {combined_output_path}")
