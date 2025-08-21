import csv

metadata_path = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/metadata_with_mutations.tsv"
fasta_in = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Alignment/sequences_pb2.fasta"
fasta_out = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Alignment/sequences_pb2_H5_only.fasta"

h5_ids = set()
with open(metadata_path, "r") as f:
    rdr = csv.DictReader(f, delimiter="\t")
    for row in rdr:
        if row.get("subtype", "").strip() == "H5":
            strain = row.get("strain", "").strip()
            if strain:
                h5_ids.add(strain)

count = 0
write_block = False
with open(fasta_in, "r") as fin, open(fasta_out, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            acc = line[1:].strip().split()[0]
            if acc in h5_ids:
                write_block = True
                fout.write(line)
                count += 1
            else:
                write_block = False
        else:
            if write_block:
                fout.write(line)

print(f"Wrote {count} H5 sequences to {fasta_out}")
