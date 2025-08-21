import pandas as pd
import os

input_csv = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_test_set.csv"
output_fasta = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Alignment/sequences_pb2.fasta"

os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

df = pd.read_csv(input_csv)
filtered_df = df[(df['label_human'] == 1) | (df['label_avian'] == 1)]

with open(output_fasta, "w") as f:
    for _, row in filtered_df.iterrows():
        accession = row["accession"]
        sequence = row["sequence"]
        f.write(f">{accession}\n{sequence}\n")

print(f"Length: {len(filtered_df)}")
