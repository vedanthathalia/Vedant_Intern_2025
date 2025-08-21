import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from datetime import datetime
import argparse
import wandb
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, choices=["human", "avian", "other"])
args = parser.parse_args()
label = args.label

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="Figures", name=f"roc_{label}_AllM_{timestamp}")

# base_model_sizes = ["8M", "35M", "650M"]
base_model_sizes = ["650M"]
embedding_base = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/embeddings_results"
fig_base_dir = f"/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/{label.capitalize()}/AllM"
os.makedirs(fig_base_dir, exist_ok=True)

def get_roc_from_xgb(X_train, y_train, X_test, y_test):
    clf = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    return fpr, tpr, auc(fpr, tpr)

sns.set_context("notebook", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, model_size in enumerate(base_model_sizes):
    print(f"\nProcessing model: {model_size}")

    ft_train_path = os.path.join(embedding_base, model_size, label, f"global_{label}_train_embeddings.csv")
    ft_test_path  = os.path.join(embedding_base, model_size, label, f"global_{label}_test_embeddings.csv")
    orig_train_path = os.path.join(embedding_base, model_size, "original", "global_original_train_embeddings.csv")
    orig_test_path  = os.path.join(embedding_base, model_size, "original", "global_original_test_embeddings.csv")

    if not (os.path.exists(ft_train_path) and os.path.exists(ft_test_path) and
            os.path.exists(orig_train_path) and os.path.exists(orig_test_path)):
        print(f"Skipping {model_size}: missing embedding files.")
        continue

    df_train_ft = pd.read_csv(ft_train_path)
    df_test_ft = pd.read_csv(ft_test_path)
    df_train_orig = pd.read_csv(orig_train_path)
    df_test_orig = pd.read_csv(orig_test_path)

    label_col = f"label_{label}"
    if label_col not in df_train_ft.columns or label_col not in df_test_ft.columns \
       or label_col not in df_train_orig.columns or label_col not in df_test_orig.columns:
        print(f"Skipping {model_size}: {label_col} column missing in one of the files.")
        continue

    X_train_ft = df_train_ft.iloc[:, 16:].values
    y_train_ft = df_train_ft[label_col].astype(int)
    X_test_ft = df_test_ft.iloc[:, 16:].values
    y_test_ft = df_test_ft[label_col].astype(int)

    X_train_orig = df_train_orig.iloc[:, 16:].values
    y_train_orig = df_train_orig[label_col].astype(int)
    X_test_orig = df_test_orig.iloc[:, 16:].values
    y_test_orig = df_test_orig[label_col].astype(int)

    print(f"X_train_ft shape: {X_train_ft.shape}")
    print(f"X_test_ft shape: {X_test_ft.shape}, X_test_orig shape: {X_test_orig.shape}")
    print(f"y_train shape: {y_train_ft.shape}, y_test shape: {y_test_ft.shape}")

    if any(x.shape[1] == 0 for x in [X_train_ft, X_test_ft, X_train_orig, X_test_orig]):
        print(f"Skipping {model_size}: one of the embeddings is empty.")
        continue

    fpr_ft, tpr_ft, auc_ft = get_roc_from_xgb(X_train_ft, y_train_ft, X_test_ft, y_test_ft)
    fpr_orig, tpr_orig, auc_orig = get_roc_from_xgb(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

    ax = axes[i]
    sns.set_context("notebook", font_scale=1.2)
    ax.plot(fpr_ft, tpr_ft, label=f"Finetuned (AUC = {auc_ft:.3f})", linewidth=2)
    ax.plot(fpr_orig, tpr_orig, label=f"Original (AUC = {auc_orig:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    # ax.set_title(f"{model_size} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)

    fig_indiv = plt.figure(figsize=(5, 5))
    plt.plot(fpr_ft, tpr_ft, label=f"Finetuned (AUC = {auc_ft:.3f})", linewidth=2)
    plt.plot(fpr_orig, tpr_orig, label=f"Original (AUC = {auc_orig:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title(f"{model_size} ROC - {label.capitalize()}")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_indiv = os.path.join(fig_base_dir, f"roc_NEW2_{label}_{model_size}.svg")
    plt.savefig(out_indiv, transparent=True)
    # wandb.log({f"{model_size} ROC": wandb.Image(out_indiv)})
    plt.close(fig_indiv)

plt.tight_layout()
combined_out = os.path.join(fig_base_dir, f"roc_NEW2_all_models_{label}.svg")
plt.savefig(combined_out, transparent=True)
# wandb.log({f"ROC Comparison All Models ({label})": wandb.Image(combined_out)})
plt.close()
wandb.finish()
