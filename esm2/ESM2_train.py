import os
import argparse
import pathlib
from datetime import date, datetime

import pandas as pd
import wandb
import rich
import torch

from datasets import Dataset as HFDataset, DatasetDict
from transformers import (
    AutoTokenizer,
    EsmForMaskedLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ESM2 model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for training")
    parser.add_argument("--label", type=str, default="human", choices=["human", "avian", "other"], help="Label group to train on")
    return parser.parse_args()

def preprocess_dataset(batch, tokenizer: PreTrainedTokenizerBase, padding, truncation, max_len):
    return tokenizer(
        batch["sequence"],
        padding=padding,
        truncation=truncation,
        max_length=max_len,
    )

def load_and_tokenize(tokenizer, label, train_config):
    base_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/"
    train_df = pd.read_csv(f"{base_path}clustered_train_{label}.csv", dtype=str).reset_index(drop=True)
    eval_df  = pd.read_csv(f"{base_path}clustered_eval_{label}.csv", dtype=str).reset_index(drop=True)
    test_df  = pd.read_csv(f"{base_path}clustered_test_{label}.csv", dtype=str).reset_index(drop=True)

    train_ds = HFDataset.from_pandas(train_df)
    eval_ds = HFDataset.from_pandas(eval_df)
    test_ds = HFDataset.from_pandas(test_df)

    ds = DatasetDict({"train": train_ds, "eval": eval_ds, "test": test_ds})

    tokenized = ds.map(
        lambda batch: preprocess_dataset(
            batch,
            tokenizer,
            train_config["padding"],
            train_config["truncation"],
            train_config["max_length"],
        ),
        batched=True,
        num_proc=min(os.cpu_count(), len(train_ds)),
        remove_columns=[c for c in train_ds.column_names if c != "sequence"],
        desc="Tokenizing dataset"
    )
    tokenized.set_format("torch")
    return tokenized, test_df


def main():
    args = parse_args()
    label = args.label

    run_name = f"esm2_finetune_{label}_{date.today().isoformat()}_{datetime.now().strftime('%H%M%S')}"
    output_dir = f"./esm_output/650M/{label}"
    logging_dir = f"./esm_logs/{label}"

    train_config = {
        "tokenizer_path": "facebook/esm2_t33_650M_UR50D",
        "model": "facebook/esm2_t33_650M_UR50D",

        "padding": "max_length",
        "truncation": False,
        "max_length": 1024,

        "fp16": True,
        "batch_size": 16,
        "num_train_epochs": 5,
        "warmup_ratio": 0.1,

        "save_strategy": "epoch",
        # "save_steps": 500,
        "save_total_limit": 5,
        "eval_strategy": "epoch",
        "eval_steps": 100,
        "logging_steps": 10,
        "logging_first_step": True,
        "overwrite_output_dir": True,

        "run_name": run_name,
        "output_dir": output_dir,
        "logging_dir": logging_dir,

        "report_to": "wandb",
        "WANDB_API_KEY": "4ccf6e6d655e18a2f2f1a5dee3114419a24080e0",
        "WANDB_PROJECT": "ESM2",
        "WANDB_RUN_GROUP": "esm_experiments",
        "WANDB_JOB_TYPE": "finetune",
        "notes": f"Finetuning ESM2 on clustered {label} dataset",
    }

    torch.manual_seed(42)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_config["tokenizer_path"])

    if train_config["report_to"] == "wandb":
        os.environ["WANDB_API_KEY"] = train_config["WANDB_API_KEY"]
        os.environ["WANDB_PROJECT"] = train_config["WANDB_PROJECT"]
        os.environ["WANDB_RUN_GROUP"] = train_config["WANDB_RUN_GROUP"]
        os.environ["WANDB_JOB_TYPE"] = train_config["WANDB_JOB_TYPE"]
        os.environ["WANDB_NOTES"] = train_config["notes"]

    tokenized_dataset, _ = load_and_tokenize(tokenizer, label, train_config)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=train_config.get("mlm_probability", 0.15),
    )

    model = EsmForMaskedLM.from_pretrained(train_config["model"])
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        run_name=train_config["run_name"],
        output_dir=train_config["output_dir"],
        logging_dir=train_config["logging_dir"],
        fp16=train_config["fp16"],
        tf32=not train_config["fp16"],
        per_device_train_batch_size=train_config["batch_size"],
        per_device_eval_batch_size=train_config["batch_size"],
        num_train_epochs=train_config["num_train_epochs"],
        save_strategy=train_config["save_strategy"],
        # save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        eval_strategy=train_config["eval_strategy"],
        eval_steps=train_config["eval_steps"],
        logging_steps=train_config["logging_steps"],
        logging_first_step=train_config["logging_first_step"],
        overwrite_output_dir=train_config["overwrite_output_dir"],
        report_to=train_config["report_to"],
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
    )

    if args.debug:
        rich.print(f"[bold blue]Train size:[/bold blue] {len(tokenized_dataset['train'])}")
        rich.print(f"[bold blue]Eval size:[/bold blue] {len(tokenized_dataset['eval'])}")
        rich.print(f"[bold blue]Test size:[/bold blue] {len(tokenized_dataset['test'])}")

    rich.print("[bold green]Starting training...[/bold green]")
    trainer.train()

    final_dir = pathlib.Path(train_config["output_dir"]) / f"final_{label}_model_5epochs"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    rich.print(f"[bold green]Model saved to {final_dir}[/bold green]")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rich.print("[bold green]Script finished.[/bold green]")


if __name__ == "__main__":
    main()

# accelerate launch ESM2_train.py --label human
# accelerate launch ESM2_train.py --label avian
# accelerate launch ESM2_train.py --label other
# accelerate launch --num_processes 1 ESM2_train.py --label avian
