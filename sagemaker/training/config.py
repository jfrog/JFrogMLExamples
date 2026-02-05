# config.py
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

JF_REPO = "llm"
JF_MODEL_NAME = "devops_helper"

MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"

# --- LoRA (PEFT) Configuration ---
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Quantization Configuration ---
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- Dataset Configuration (fixed) ---
DATASET_ID = "Szaid3680/Devops"

# Max sequence length might be model-specific but rarely tuned
MAX_SEQ_LENGTH = 256


# Qwen1.5 (on CPU)
# --- Prompt Engineering ---
def get_prompt(text: str) -> str:
    """
    Creates a simple default prompt for fallback text-only datasets.
    """
    system_prompt = "You are a helpful DevOps assistant."
    instruction = f"{text}"

    # Qwen1.5 uses ChatML format
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )