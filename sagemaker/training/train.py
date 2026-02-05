# training/train.py

import os
import base64

import boto3
import torch

from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import frogml
import argparse

import config   # <-- import config.py
import dataset_utils

from huggingface_hub import snapshot_download
from trl import SFTTrainer
from peft import get_peft_model

def parse_args():
    parser = argparse.ArgumentParser()

    # --- Model & Dataset ---
    parser.add_argument("--dataset_sample_percentage", type=float, default=1.0)
    
    # --- Training Hyperparameters ---
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=1)

    return parser.parse_args()


def _get_secret_id(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def _get_secret_value(secret_id: str) -> str:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_id)
    secret = response.get("SecretString")
    if secret is None:
        secret = base64.b64decode(response["SecretBinary"]).decode("utf-8")
    return secret


def main():
    
    args = parse_args()

    hf_token_secret_id = _get_secret_id("HF_TOKEN_SECRET_ID")
    os.environ["HF_TOKEN"] = _get_secret_value(hf_token_secret_id)


    jf_token_secret_id = _get_secret_id("JF_ACCESS_TOKEN_SECRET_ID")
    os.environ["JF_ACCESS_TOKEN"] = _get_secret_value(jf_token_secret_id)

    local_path = snapshot_download(repo_id=config.MODEL_ID)

    print("Model downloaded to:", local_path)


    # Load model from local path
    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else "cpu"
    quantization_config = None
    use_fp16 = False
    use_bf16 = False
    
    if use_cuda:
        # Use 4-bit quantization by default on GPU to reduce memory usage.
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("✅ CUDA detected. Configuring mixed precision.")
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True
    else:
        print("⚠️ No CUDA detected. Mixed precision flags (fp16/bf16) will be disabled.")

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # Apply LoRA configuration to the model
    model = get_peft_model(model, config.LORA_CONFIG)

    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(local_path)


    # Using left-padding with the beginning-of-sentence token is more robust.
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = config.MAX_SEQ_LENGTH
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    # 2. Load and Tokenize Dataset
    train_dataset, eval_dataset, format_instruction = dataset_utils.load_datasets(
        percentage=args.dataset_sample_percentage
    )

    # 3. Configure and run the Hugging Face Trainer
    adapter_output_dir = os.environ.get("SM_MODEL_DIR", "./output")

    training_args = TrainingArguments(
        output_dir=adapter_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        max_steps=args.max_steps,
        logging_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #peft_config=LORA_CONFIG,
        formatting_func=format_instruction,
        args=training_args,
    )

    # 4. Start Training and Log Artifacts
    print("Starting model training...")
    train_output = trainer.train()
    print("Training complete.")
    
    eval_metrics = trainer.evaluate()
    print("--- Evaluation Metrics ---")
    print(eval_metrics)

    # Log all hyperparameters used for the training run
    params_to_log = {
        "model_id": config.MODEL_ID,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.train_batch_size,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "lora_r": config.LORA_CONFIG.r,
        "lora_alpha": config.LORA_CONFIG.lora_alpha,
        #"lora_target_modules": str(config.LORA_CONFIG.target_modules),
    }
 

    final_metrics = train_output.metrics | eval_metrics

    print(f"Logged Parameters: {params_to_log}")
    print(f"Logged Metrics: {final_metrics}")

    try:

        model_to_log = trainer.model
        if hasattr(model_to_log, "merge_and_unload"):
            model_to_log = model_to_log.merge_and_unload()

        model_to_log.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)

        frogml.huggingface.log_model(
            model=model_to_log,
            tokenizer=tokenizer,
            repository=config.JF_REPO,    # The JFrog repository to upload the model to.
            model_name=config.JF_MODEL_NAME,     # The uploaded model name
            version="",     # Optional. Defaults to timestamp
            parameters=params_to_log | {"finetuning-dataset": config.DATASET_ID},
            metrics = final_metrics,
            )

        print("--- Model Logged to JFrogML Successfully ---")

    except Exception as e:
        print(f"An error occurred during model logging: {e}")


if __name__ == "__main__":
    main()
