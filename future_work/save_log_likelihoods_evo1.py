import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_paths(host_type):
    base_dir = "/home/vhathalia/SCRIPPS_jupyter/EVO_Model"
    model_path = f"{base_dir}/evo1_finetuned/final_model"
    csv_path = f"{base_dir}/NCBI_Dataset/clustered_test_{host_type}.csv"
    out_path = f"{base_dir}/log_likelihood_results/evo1/{host_type}_log_likelihoods.csv"
    return model_path, csv_path, out_path

def calculate_batch_log_likelihood(model, tokenizer, sequences, device):
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=model.config.max_position_embeddings,
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    batch_size = input_ids.size(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attention = attention_mask[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    ).view(batch_size, -1)

    log_likelihoods = - (losses * shift_attention).sum(dim=1)
    return log_likelihoods.cpu().tolist()


def compute_log_likelihoods(model, tokenizer, sequences, device, batch_size=32):
    all_ll = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Computing"):
        batch = sequences[i:i + batch_size]
        batch_ll = calculate_batch_log_likelihood(model, tokenizer, batch, device)
        all_ll.extend(batch_ll)
    return all_ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--avian", action="store_true")
    parser.add_argument("--other", action="store_true")
    args = parser.parse_args()

    if args.human:
        host = "human"
    elif args.avian:
        host = "avian"
    elif args.other:
        host = "other"

    model_path, csv_path, output_csv = get_paths(host)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = "X"
    model.eval()

    df = pd.read_csv(csv_path)
    df["log_likelihood"] = np.nan

    sequences = df["sequence"].dropna().astype(str).tolist()
    valid_indices = df["sequence"].dropna().index.tolist()

    ll_scores = compute_log_likelihoods(model, tokenizer, sequences, device)
    df.loc[valid_indices, "log_likelihood"] = ll_scores
    df.to_csv(output_csv, index=False)
    print(f"\nSaved log-likelihoods to: {output_csv}")


if __name__ == "__main__":
    main()

# python save_log_likelihoods.py --human
# python save_log_likelihoods.py --avian
# python save_log_likelihoods.py --other
