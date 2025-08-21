import pandas as pd
from collections import defaultdict

full_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/EVO_Model/data80_clu_vectorized.tsv", sep="\t")

def split_dataset(label_col, label_name, train_target=0.8):
    print(f"\n{label_name.upper()}")
    
    label_df = full_df[full_df[label_col] == 1].copy()
    cluster_counts = label_df["cluster_id"].value_counts().reset_index()
    cluster_counts.columns = ["cluster_id", "count"]
    
    # Sort by largest cluster first (descending)
    cluster_counts = cluster_counts.sort_values(by="count", ascending=False).reset_index(drop=True)

    total_seqs = len(label_df)
    train_seqs, train_clusters = 0, []

    for _, row in cluster_counts.iterrows():
        cluster_id, count = row["cluster_id"], row["count"]
        if train_seqs / total_seqs >= train_target and len(train_clusters) >= 2:
            break
        train_seqs += count
        train_clusters.append(cluster_id)

    train_df = label_df[label_df["cluster_id"].isin(train_clusters)].reset_index(drop=True)
    eval_df = label_df[~label_df["cluster_id"].isin(train_clusters)].reset_index(drop=True)

    # Define test set as everything else (not in train or eval) + eval
    train_eval_ids = set(train_df["accession"]).union(set(eval_df["accession"]))
    test_df = pd.concat([full_df[~full_df["accession"].isin(train_eval_ids)], eval_df], ignore_index=True)

    # Save CSVs
    base = "/home/vhathalia/SCRIPPS_jupyter/EVO_Model/NCBI_Dataset/"
    train_df.to_csv(f"{base}clustered_train_{label_name}.csv", index=False)
    eval_df.to_csv(f"{base}clustered_eval_{label_name}.csv", index=False)
    test_df.to_csv(f"{base}clustered_test_{label_name}.csv", index=False)

    # Print stats
    print(f"Train sequences: {len(train_df)} ({round(100 * len(train_df)/total_seqs, 2)}%)")
    print(f"Eval sequences: {len(eval_df)} ({round(100 * len(eval_df)/total_seqs, 2)}%)")
    print(f"Train clusters: {train_df['cluster_id'].nunique()}")
    print(f"Eval clusters: {eval_df['cluster_id'].nunique()}")
    print(f"Test sequences: {len(test_df)}")

# Run for each label
split_dataset("label_human", "human", train_target=0.8)
split_dataset("label_avian", "avian", train_target=0.8)
split_dataset("label_other", "other", train_target=0.8)

# Output files:
# clustered_train_human.csv, clustered_eval_human.csv, clustered_test_human.csv
# clustered_train_avian.csv, clustered_eval_avian.csv, clustered_test_avian.csv
# clustered_train_other.csv, clustered_eval_other.csv, clustered_test_other.csv

# /home/vhathalia/SCRIPPS_jupyter/EVO_Model/NCBI_Dataset/clustered_train_human.csv
# /home/vhathalia/SCRIPPS_jupyter/EVO_Model/NCBI_Dataset/clustered_eval_human.csv
# /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_human.csv