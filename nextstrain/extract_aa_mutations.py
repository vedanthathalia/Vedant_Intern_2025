import csv
from pathlib import Path
from Bio import SeqIO

ALIGNED_PATH = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Alignment/aligned_pb2.fasta"
METADATA_PATH = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/metadata.tsv"
REFERENCE_ID = "AAD51922.1"
OUTPUT_PATH = "metadata_with_mutations.tsv"

def load_metadata(metadata_path):
    rows = {}
    with open(metadata_path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        id_cols = [c for c in ("strain", "accession", "name", "seq_id") if c in rdr.fieldnames]
        if not id_cols or "host" not in rdr.fieldnames:
            raise SystemExit("metadata.tsv must have a 'host' column and one of: strain/accession/name/seq_id")
        id_col = id_cols[0]
        for r in rdr:
            rows[r[id_col]] = r
        return rows, rdr.fieldnames, id_col

def main():
    metadata_rows, fieldnames, id_col = load_metadata(METADATA_PATH)

    if "mutations" not in fieldnames:
        fieldnames.append("mutations")

    records = list(SeqIO.parse(ALIGNED_PATH, "fasta"))
    if not records:
        raise SystemExit("No sequences found in aligned FASTA.")

    ref_rec = None
    for rec in records:
        if rec.id == REFERENCE_ID:
            ref_rec = rec
            break
    if ref_rec is None:
        raise SystemExit(f"Reference id '{REFERENCE_ID}' not found in aligned FASTA.")

    ref_seq = str(ref_rec.seq)
    aln_len = len(ref_seq)

    ref_pos_map = []
    pos = 0
    for c in ref_seq:
        if c != "-":
            pos += 1
            ref_pos_map.append(pos)
        else:
            ref_pos_map.append(None)

    for rec in records:
        muts = []
        if rec.id != ref_rec.id:
            seq = str(rec.seq)
            for i in range(aln_len):
                r = ref_seq[i]
                a = seq[i]
                pos_i = ref_pos_map[i]
                if pos_i is None:
                    continue
                if r == "-" or a == "-" or r.upper() == "X" or a.upper() == "X":
                    continue
                if a != r:
                    muts.append(f"{r}{pos_i}{a}")
        if rec.id in metadata_rows:
            metadata_rows[rec.id]["mutations"] = ", ".join(muts)
        else:
            print(f"Warning: {rec.id} in alignment but not in metadata.tsv")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in metadata_rows.values():
            if "mutations" not in row:
                row["mutations"] = ""
            w.writerow(row)

    print(f"Done. Wrote {OUTPUT_PATH}")
    print(f"Reference: {ref_rec.id}")
    print(f"Sequences processed: {len(records)}")

if __name__ == "__main__":
    main()
