import pandas as pd
from datasets import Dataset

label_group = "human" # avian, other

base_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/"
train_csv_path = f"{base_path}clustered_train_{label_group}.csv"
eval_csv_path  = f"{base_path}clustered_eval_{label_group}.csv"
test_csv_path  = f"{base_path}clustered_test_{label_group}.csv"

train_df = pd.read_csv(train_csv_path).reset_index(drop=True)
eval_df  = pd.read_csv(eval_csv_path).reset_index(drop=True)
test_df  = pd.read_csv(test_csv_path).reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
eval_dataset  = Dataset.from_pandas(eval_df)
test_dataset  = Dataset.from_pandas(test_df)

__all__ = ["train_dataset", "eval_dataset", "test_dataset"]
