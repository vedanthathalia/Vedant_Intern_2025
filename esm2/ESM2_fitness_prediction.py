"""Inspired by https://doi.org/10.1101/2025.05.20.655154 and https://huggingface.co/blog/AmelieSchreiber/mutation-scoring"""

from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import re

# Constants (if needed globally, otherwise pass as arguments or define in main)
# HUMAN_REF = "MKTIIALSYILCLVFAQKLPGNDNSTATLCLGHHAVPNGTIVKTITNDQIEVTNATELVQSSSTGGICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNDESFNWTGVTQNGTSSSCKRRSNNSFFSRLNWLTHLKFKYPALNVTMPNNEKFDKLYIWGVHHPVTDNDQIFLYAQASGRITVSTKRSQQTVIPNIGSRPRIRNIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCNSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGIGQAADLKSTQAAINQINGKLNRLIGKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFERTKKQLRENAEDMGNGCFKIYHKCDNACIGSIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVALLGFIMWACQKGNIRCNICI"
# TEST_SEQ = "MNTRILILTLAAVTHTNADKICLGHHAVANGTKVNTLTERGVEVVNATETVEQTNIPRICTKGKKAIDLGQCGLLGIITGPPQCDQFLEFTADLIIERREGNDVCYPGKFVNEEALRQILRESGGINKETTGFTYSGIRTNGVTSACRRSGSSFYAEMKWLLSNTDNAAFPQMTKSYKNTRNEPALIVWGIHHSGSAAEQTKLYGSGNKLITVGSSNYQQSFVPSPGARPQVNGQSGRIDFHWLILNPNDTVTFSFNGAFVAPDRVSFFKGKSTGIQSEVPVDINCEGECYHSGGTITSNLPFQNVNSRAVGKCPRYVKQKSLLLATGMKNVPEIPKKKKRGLFGAIAGFIENGWEGLVDGWYGFRHQNAQGEGTAADYKSTQSAIDQITGKLNRLIEKTNQQFELIDNEFTEVEKQIGNVINWTRDSITEVWSYNAELLVAMENQHTIDLADSEMNKLYERVRRQLRENAEEDGTGCFEIFHKCDDDCMASIRNNTYDHSTYREEAMQNRVKIDPVKLSSGYKDVILWFSFGASCFLLLAIAMGLVFICVKNGNMRCTICI"


def calculate_batch_log_likelihood_from_inputs(model, tokenized_inputs):
    """
    Calculates log-likelihoods for a batch of tokenized sequences.
    Assumes tokenized_inputs contains 'input_ids' and 'attention_mask'
    and is already on the correct device.
    """
    input_ids = tokenized_inputs.input_ids
    attention_mask = tokenized_inputs.attention_mask

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    log_probs_full = outputs.logits.log_softmax(dim=-1)
    log_probs_true_tokens = torch.gather(
        log_probs_full, dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    masked_log_probs = log_probs_true_tokens * attention_mask
    summed_log_likelihoods = masked_log_probs.sum(dim=1)

    return summed_log_likelihoods.cpu().tolist()

def setup_paths_and_dirs(model_name, output_file_basename):
    out_dir = Path("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow user to pass "subfolder/filename.csv" as output_file_basename
    output_path = out_dir / output_file_basename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return str(output_path)


def load_model_and_tokenizer(model_name, device):
    """Loads ESM model and tokenizer and moves model to device."""
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_and_prepare_data(csv_file_path):
    """Loads data from CSV and prepares sequences for processing."""
    df = pd.read_csv(csv_file_path)
    df["log_likelihood"] = np.nan

    valid_sequences = []
    valid_indices = []
    for i, row in df.iterrows():
        seq = row["sequence"]
        if pd.isna(seq):
            raise ValueError(
                f"Sequence at original DataFrame index {i} is NaN. Please check your input data."
            )
        valid_sequences.append(str(seq))
        valid_indices.append(i)
    return df, valid_sequences, valid_indices


def calculate_log_likelihoods_for_sequences(
    model, tokenizer, sequences, device, batch_size=256
):
    """Calculates log-likelihoods for a list of sequences in batches."""
    all_calculated_log_likelihoods = []
    if not sequences:
        return all_calculated_log_likelihoods

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch_seq_list = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch_seq_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings - 2,
        ).to(device)
        log_likelihoods_batch = calculate_batch_log_likelihood_from_inputs(
            model, inputs
        )
        all_calculated_log_likelihoods.extend(log_likelihoods_batch)
    return all_calculated_log_likelihoods

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <csv_file> <output_file_basename>")
        sys.exit(1)

    model_name_arg = sys.argv[1]
    csv_file_arg = sys.argv[2]
    output_file_basename_arg = sys.argv[3]

    output_file_prefix = setup_paths_and_dirs(model_name_arg, output_file_basename_arg)
    print(f"Output file prefix set to: {output_file_prefix}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(model_name_arg, device)
    print(f"Model '{model_name_arg}' and tokenizer loaded.")

    df, valid_sequences, valid_indices = load_and_prepare_data(csv_file_arg)
    print(
        f"Data loaded from '{csv_file_arg}'. Found {len(valid_sequences)} valid sequences."
    )

    if valid_sequences:
        log_likelihoods = calculate_log_likelihoods_for_sequences(
            model, tokenizer, valid_sequences, device
        )
        df.loc[valid_indices, "log_likelihood"] = log_likelihoods
        print("Log-likelihoods calculated.")
    else:
        print("No valid sequences to process.")

    df.to_csv(output_file_prefix + ".csv", index=False)
    print(f"Results saved to {output_file_prefix}.csv")


if __name__ == "__main__":
    main()

# python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/8M/human/final_human_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_human.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/8M/final_human_model/human_log_likelihoods
# python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/35M/human/final_human_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_human.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/35M/final_human_model/human_log_likelihoods

# Avian
# DONE - python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/8M/avian/final_avian_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_avian.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/8M/final_avian_model/avian_log_likelihoods
# DONE - python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/35M/avian/final_avian_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_avian.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/35M/final_avian_model/avian_log_likelihoods

# Other
# DONE - python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/8M/other/final_other_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_other.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/8M/final_other_model/other_log_likelihoods
# DONE - python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/35M/other/final_other_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/clustered_test_other.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/35M/final_other_model/other_log_likelihoods


# python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/650M/avian/final_avian_model_5epochs /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_test_set.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_avian_model_5epochs/global_test_log_likelihoods
# python ESM2_fitness_prediction.py /home/vhathalia/SCRIPPS_jupyter/ESM_Model/esm_output/650M/human/final_human_model /home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_test_set.csv /home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_human_model/global_test_log_likelihoods