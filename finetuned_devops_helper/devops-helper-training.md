# 📓 DevOps Helper Training - Local Notebook Guide

This guide shows you how to **fine-tune an LLM locally** for DevOps question answering and **log it to JFrog ML Registry** for later deployment.

## 🎯 Overview

**Workflow**: Local Fine-tuning → JFrog ML Registry → Production Deployment

**Technology**: Fine-tuned **Qwen 1.5 0.5B** model using **LoRA** (Low-Rank Adaptation)
**Dataset**: DevOps instruction dataset for domain-specific responses
**Method**: Parameter-efficient fine-tuning with minimal computational requirements

---

## 📋 Prerequisites

- **Python Environment**: 3.9-3.11 with required packages
- **JFrogML CLI**: Configured and authenticated
- **Hugging Face Access**: For model downloads

---

## 📓 Notebook Instructions

Create a new Jupyter notebook and follow these cells:

### **Cell 1: Introduction (Markdown)**
```markdown
# Fine-Tuning a Qwen 1.5 Model for DevOps Q&A

This notebook demonstrates the process of fine-tuning a small-scale Qwen model (`Qwen/Qwen1.5-0.5B-Chat`) on a DevOps instruction dataset. We use Parameter-Efficient Fine-Tuning (PEFT) with LoRA for memory efficiency.

**Key Steps:**
1. **Setup**: Install libraries and import modules
2. **Configuration**: Define model, dataset, and training parameters  
3. **Data Preparation**: Load and format DevOps dataset
4. **Model Fine-Tuning**: Apply LoRA and train with SFTTrainer
5. **Evaluation**: Compare base vs fine-tuned model responses
6. **Registry Logging**: Save model to JFrog ML Registry
```

### **Cell 2: Setup and Imports (Code)**
```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
import frogml  # JFrogML integration for model logging
```

### **Cell 3: Model Configuration (Markdown)**
```markdown
## Configuration

Define all training parameters, model settings, and LoRA configuration for the fine-tuning process.
```

### **Cell 4: Configuration Parameters (Code)**
```python
# Model and tokenizer configuration
model_id = "Qwen/Qwen1.5-0.5B-Chat"
new_model_adapter = "qwen-0.5b-devops-adapter"

# Dataset configuration
dataset_name = "Szaid3680/Devops"

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,                    # Low-rank dimension
    lora_alpha=32,           # LoRA scaling parameter
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments - optimized for quick demo
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=10,           # Quick demo - only 10 steps
    fp16=False,             # CPU/MPS compatibility
)
```

### **Cell 5: Data Loading and Preparation (Markdown)**
```markdown
## Data Preparation

Load the DevOps instruction dataset and format it for fine-tuning. We'll use a small subset for this demo.
```

### **Cell 6: Dataset Loading (Code)**
```python
# Load DevOps dataset
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Quick demo - use small subset
train_dataset = train_dataset.select(range(10))  # 10 samples for demo
eval_dataset = eval_dataset.select(range(5))     # 5 samples for evaluation

def format_instruction(example):
    """Format dataset examples into structured prompts for instruction tuning."""
    instruction = example.get('Instruction', '')
    inp = example.get('Prompt', '')
    response = example.get('Response', '')
    
    full_prompt = f"<s>[INST] {instruction}\n{inp} [/INST] {response} </s>"
    return full_prompt

# Preview sample data
print("Sample DevOps training example:")
print(f"Instruction: {train_dataset[0]['Instruction'][:100]}...")
print(f"Prompt: {train_dataset[0]['Prompt'][:100]}...")
print(f"Response: {train_dataset[0]['Response'][:100]}...")
```

### **Cell 7: Model Loading and Fine-Tuning Setup (Markdown)**
```markdown
## Model Loading and Fine-Tuning

Load the base Qwen model, apply LoRA configuration, and set up the training pipeline.
```

### **Cell 8: Model Setup and Training (Code)**
```python
# Force CPU usage for compatibility
device = "cpu"
print(f"🎯 Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu"  # Force CPU for demo compatibility
)

# Apply LoRA configuration
model = get_peft_model(model, lora_config)
model = model.to(device)

# Create SFTTrainer for instruction fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=format_instruction,
    args=training_args,
)

print("🚀 Starting Fine-Tuning...")
trainer.train()
print("✅ Fine-Tuning Complete!")
```

### **Cell 9: Model Evaluation (Markdown)**
```markdown
## Model Evaluation

Evaluate the fine-tuned model and compare its DevOps responses against the base model.
```

### **Cell 10: Evaluation and Comparison (Code)**
```python
# Evaluate fine-tuned model
metrics = trainer.evaluate()
print("📊 Evaluation Metrics:")
print(f"Eval Loss: {metrics['eval_loss']:.4f}")

# Save the LoRA adapter
trainer.model.save_pretrained(new_model_adapter)

# Merge adapter with base model for inference
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
finetuned_model = PeftModel.from_pretrained(base_model, new_model_adapter)
finetuned_model = finetuned_model.merge_and_unload()

# Test DevOps question
prompt = "How do I expose a deployment in Kubernetes using a service?"
messages = [
    {"role": "system", "content": "You are a helpful DevOps assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# Generate responses for comparison
model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
input_ids_len = model_inputs['input_ids'].shape[1]

print("🤖 FINE-TUNED MODEL RESPONSE:")
print("-" * 50)
generated_ids = finetuned_model.generate(model_inputs.input_ids, max_new_tokens=100)
response_only_ids = generated_ids[:, input_ids_len:]
response_finetuned = tokenizer.decode(response_only_ids[0], skip_special_tokens=True)
print(response_finetuned)

print("\n📚 BASE MODEL RESPONSE:")
print("-" * 50)
original_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
generated_ids_base = original_model.generate(model_inputs.input_ids, max_new_tokens=100)
response_only_ids_base = generated_ids_base[:, input_ids_len:]
response_base = tokenizer.decode(response_only_ids_base[0], skip_special_tokens=True)
print(response_base)
```

### **Cell 11: Save to JFrog ML Registry (Markdown)**
```markdown
## Save to JFrog ML Registry

Log the fine-tuned DevOps helper model to JFrog ML Registry with all metadata and dependencies.
```

### **Cell 12: Registry Logging (Code)**
```python
# IMPORTANT: Update this path to your actual project location
base_projects_directory = "/Users/your-username/Projects/JFrogMLExamples"  # Update this path!

try:
    # Log model to JFrog ML Registry
    frogml.huggingface.log_model(   
        model=finetuned_model,
        tokenizer=tokenizer,
        repository="llm",                    # JFrog repository name
        model_name="devops_helper",          # Model name in registry
        version="",                          # Auto-generated version
        parameters={
            "finetuning-dataset": dataset_name,
            "model_base": model_id,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "max_steps": training_args.max_steps
        },
        code_dir=f"{base_projects_directory}/finetuned_devops_helper/code_dir",
        dependencies=[f"{base_projects_directory}/finetuned_devops_helper/main/conda.yaml"],
        metrics=metrics,
        predict_file=f"{base_projects_directory}/finetuned_devops_helper/code_dir/predict.py"
    )
    print("✅ Model Logged Successfully to JFrog ML Registry!")
    print("🎯 Model Name: devops_helper")
    print("📦 Repository: llm")
    
except Exception as e:
    print(f"❌ Error during model logging: {e}")
    print("💡 Make sure your JFrogML CLI is configured and authenticated")
```

### **Cell 13: Next Steps (Markdown)**
```markdown
## Next Steps

**Model logged to JFrog ML Registry** → Ready for deployment

**Production deployment**: [Remote Training & Deployment Guide](remote-training-and-deployment.md)
```

---

## 📝 Usage Instructions

1. **Create** a new Jupyter notebook
2. **Copy** each cell content (13 cells total: 7 Markdown + 6 Code)
3. **Update** the `base_projects_directory` path in Cell 12
4. **Run** cells sequentially

**Training Time**: ~2-3 minutes on CPU
