import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load pre-merged overlapping data
merged = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/overlapping_sequences.csv")

# Human Sequences Plot
human_seqs = merged[merged['label_human'] == 1].copy()
# both_labels = merged[(merged['label_human'] == 1) & (merged['label_avian'] == 1)].copy()
# human_seqs['quartile'] = pd.qcut(human_seqs['log_likelihood_human'], 4, labels=["Q1", "Q2", "Q3", "Q4"])

# human_seqs = human_seqs[human_seqs['quartile'] == "Q4"]
# human_seqs = human_seqs[human_seqs["log_likelihood_avian"] > -75]

# human_seqs = human_seqs[human_seqs["log_likelihood_avian"] > human_seqs["log_likelihood_avian"].quantile(0.75)]
# human_seqs = human_seqs[human_seqs["log_likelihood_avian"] < human_seqs["log_likelihood_avian"].quantile(1.0)]

# human_seqs = human_seqs[human_seqs["log_likelihood_human"] > human_seqs["log_likelihood_human"].quantile(0.75)]
# human_seqs = human_seqs[human_seqs["log_likelihood_human"] < human_seqs["log_likelihood_human"].quantile(1.0)]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=human_seqs, x='log_likelihood_human', y='log_likelihood_avian', color='blue')
sns.regplot(data=human_seqs, x='log_likelihood_human', y='log_likelihood_avian', scatter=False, color='black', line_kws={"linestyle": "--"})
plt.title("Human Sequences: Human vs Avian Log-Likelihoods")
plt.xlabel("Log-Likelihood (Human Model)")
plt.ylabel("Log-Likelihood (Avian Model)")

plt.tight_layout()
plt.savefig("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Regression/regression_human_sequences_Q2toQ4.png")

# Avian Sequences Plot
avian_seqs = merged[merged['label_avian'] == 1].copy()
# both_labels = merged[(merged['label_human'] == 1) & (merged['label_avian'] == 1)].copy()
# avian_seqs['quartile'] = pd.qcut(avian_seqs['log_likelihood_avian'], 4, labels=["Q1", "Q2", "Q3", "Q4"])

# avian_seqs = avian_seqs[avian_seqs['quartile'] == "Q4"]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=avian_seqs, x='log_likelihood_avian', y='log_likelihood_human', color='red')
sns.regplot(data=avian_seqs, x='log_likelihood_avian', y='log_likelihood_human', scatter=False, color='black', line_kws={"linestyle": "--"})
plt.title("Avian Sequences: Avian vs Human Log-Likelihoods")
plt.xlabel("Log-Likelihood (Avian Model)")
plt.ylabel("Log-Likelihood (Human Model)")

plt.tight_layout()
plt.savefig("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Regression/regression_avian_sequences_Q2toQ4.png")
