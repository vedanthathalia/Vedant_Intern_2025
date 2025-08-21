import os
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def save_representatives_by_subtype(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    df = df[(df["label_avian"] == 1) | (df["label_human"] == 1)]

    rep_df = df[df["accession"] == df["cluster_id"]]

    os.makedirs(output_dir, exist_ok=True)

    subtypes = rep_df['HA_subtype'].dropna().unique()

    for subtype in subtypes:
        subtype_df = rep_df[rep_df['HA_subtype'] == subtype]

        records = [
            SeqRecord(Seq(row.sequence), id=row.accession, description="")
            for row in subtype_df.itertuples(index=False)
        ]

        fasta_path = os.path.join(output_dir, f"representatives_{subtype}.fasta")
        SeqIO.write(records, fasta_path, "fasta")
        print(f"Saved {len(records)} sequences for subtype {subtype} to {fasta_path}")

    print(f"Total unique subtypes saved: {len(subtypes)}")

    all_records = [
        SeqRecord(Seq(row.sequence), id=row.accession, description="")
        for row in rep_df.itertuples(index=False)
    ]

    SeqIO.write(all_records, "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/representatives_all.fasta", "fasta")
    print(f"Saved all HA subtype sequences to representatives_all.fasta")


if __name__ == "__main__":
    input_csv = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_test_set.csv"
    output_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/representatives_by_subtype"
    save_representatives_by_subtype(input_csv, output_dir)

# mafft /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/representatives_by_subtype/representatives_HX.fasta > /home/vhathalia/SCRIPPS_jupyter/ESM_Model/mafft_output/HX_pb2_aligned.fasta

