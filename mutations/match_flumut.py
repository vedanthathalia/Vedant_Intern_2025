import pandas as pd

mutation_csv = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/new_clade_analysis_mutations.csv"
flumut_mutations_csv = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/flumut_mutations_input.csv"
output_csv = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Mutations/compared_mutations_output.csv"

df = pd.read_csv(mutation_csv)
df["Parsed_Mutations"] = df["Mutations"].fillna("").apply(
    lambda x: [m.strip() for m in x.split(";") if m.strip()]
)

flumut_df = pd.read_csv(flumut_mutations_csv)
flumut_set = set(flumut_df["Mutation"].dropna().str.strip())

def match_mutations(muts):
    if not muts:
        return "N/A"
    found = [m for m in muts if m in flumut_set]
    return "; ".join(found) if found else "N/A"

df["Matched_FluMut_Mutations"] = df["Parsed_Mutations"].apply(match_mutations)

df[["Sequence_ID", "Matched_FluMut_Mutations"]].to_csv(output_csv, index=False)
print(f"Output saved to: {output_csv}")
