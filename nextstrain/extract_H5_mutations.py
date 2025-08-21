import csv
from pathlib import Path
from Bio import SeqIO

ALIGNED_PATH = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Alignment/aligned_pb2_H5_only.fasta"
METADATA_PATH = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/metadata_with_mutations.tsv"
FLUMUT_LIST_PATH = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/Flumut/flumut_mutations_input.csv"
REFERENCE_ID = "AAD51922.1"
OUTPUT_PATH = "/home/vhathalia/SCRIPPS_jupyter/nextstrain_pb2/H5_metadata_with_mutations.tsv"

def load_metadata_all(metadata_path):
    rows = {}
    with open(metadata_path, newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        if rdr.fieldnames is None:
            raise SystemExit("Could not read header from metadata.tsv")
        id_cols = [c for c in ("strain", "accession", "name", "seq_id") if c in rdr.fieldnames]
        if not id_cols:
            raise SystemExit("metadata.tsv must have one of: strain/accession/name/seq_id")
        if "subtype" not in rdr.fieldnames:
            raise SystemExit("metadata.tsv must have a 'subtype' column")
        id_col = id_cols[0]

        for r in rdr:
            key = (r.get(id_col) or "").strip()
            if key:
                rows[key] = r
        return rows, rdr.fieldnames, id_col

def load_flumut_set(csv_path):
    muts = set()
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        if "Mutation" not in rdr.fieldnames:
            raise SystemExit("flumut_mutations_input.csv must have a 'Mutation' column")
        for r in rdr:
            m = (r.get("Mutation") or "").strip()
            if m:
                muts.add(m)
    return muts

def main():
    metadata_rows, orig_fieldnames, id_col = load_metadata_all(METADATA_PATH)

    h5_ids = {k for k, v in metadata_rows.items() if (v.get("subtype") or "").strip() == "H5"}

    fieldnames = list(orig_fieldnames)
    if "mutations" not in fieldnames:
        fieldnames.append("mutations")
    if "human_adaptive_mutations" not in fieldnames:
        insert_at = fieldnames.index("mutations") + 1 if "mutations" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "human_adaptive_mutations")

    records = list(SeqIO.parse(ALIGNED_PATH, "fasta"))
    if not records:
        raise SystemExit("No sequences found in aligned FASTA.")

    ref_rec = next((rec for rec in records if rec.id == REFERENCE_ID), None)
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
        if rec.id == ref_rec.id:
            if rec.id in metadata_rows and rec.id in h5_ids:
                metadata_rows[rec.id]["mutations"] = ""
            continue

        if rec.id in metadata_rows and rec.id in h5_ids:
            seq = str(rec.seq)
            muts = []
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
            metadata_rows[rec.id]["mutations"] = ", ".join(muts)

    flumut_set = load_flumut_set(FLUMUT_LIST_PATH)

    for k, row in metadata_rows.items():
        if k in h5_ids:
            row_muts_raw = row.get("mutations", "") or ""
            row_muts = {m.strip() for m in row_muts_raw.split(",") if m.strip()}
            overlap = sorted(row_muts & flumut_set, key=lambda x: (x[1:-1].isdigit(), x))
            row["human_adaptive_mutations"] = ", ".join(overlap)
        else:
            if "human_adaptive_mutations" not in row:
                row["human_adaptive_mutations"] = row.get("human_adaptive_mutations", "")

        if "mutations" not in row:
            row["mutations"] = row.get("mutations", "")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in metadata_rows.values():
            w.writerow(row)

    print(f"Done. Wrote {OUTPUT_PATH}")
    print(f"Reference in alignment: {ref_rec.id}")
    print(f"Sequences in alignment: {len(records)}")
    print(f"H5 rows in metadata: {len(h5_ids)}")

if __name__ == "__main__":
    main()
