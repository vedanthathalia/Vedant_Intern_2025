import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
from datasets import Dataset as HFDataset
from tqdm import tqdm

def save_mean_embeddings(
    model_path, tokenizer_path, csv_file, save_path, max_length=1024
):
    device = torch.device("cuda")
    model = EsmForMaskedLM.from_pretrained(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = pd.read_csv(csv_file)
    ds = HFDataset.from_pandas(df)

    tokenized_ds = ds.map(
        lambda batch: tokenizer(batch["sequence"], padding="max_length", truncation=False, max_length=max_length),
        batched=True
    )
    tokenized_ds.set_format("torch")

    all_embs = []
    for ex in tqdm(tokenized_ds, desc="Embedding Sequences"):
        inputs = {
            "input_ids": ex["input_ids"].unsqueeze(0).to(device),
            "attention_mask": ex["attention_mask"].unsqueeze(0).to(device)
        }
        with torch.no_grad():
            out = model.esm(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            all_embs.append(emb)

    emb_df = pd.DataFrame(all_embs)
    full_df = pd.read_csv(csv_file)
    full_df.reset_index(drop=True, inplace=True)
    emb_df.columns = [f"embedding_{i}" for i in range(emb_df.shape[1])]
    merged_df = pd.concat([full_df, emb_df], axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    base_input = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_test_set.csv"
    base_output = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/embeddings_results/8M"
    tokenizer_path = "facebook/esm2_t6_8M_UR50D"

    configs = [
        # {
        #     "name": "avian",
        #     "model": "./esm_output/35M/avian/final_avian_model",
        #     "out": f"{base_output}/avian/global_avian_train_embeddings.csv",
        # },
        {
            "name": "other",
            "model": "./esm_output/8M/other/final_other_model",
            "out": f"{base_output}/other/global_other_test_embeddings.csv",
        },
        # {
        #     "name": "original",
        #     "model": "facebook/esm2_t12_35M_UR50D",
        #     "out": f"{base_output}/original/global_original_train_embeddings.csv",
        # },
        # {
        #     "name": "human",
        #     "model": "./esm_output/35M/human/final_human_model",
        #     "out": f"{base_output}/human/global_human_train_embeddings.csv",
        # },
    ]

    for cfg in configs:
        print(f"Processing {cfg['name']}")
        save_mean_embeddings(cfg["model"], tokenizer_path, base_input, cfg["out"])
