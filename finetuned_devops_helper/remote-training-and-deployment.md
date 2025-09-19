# 🚀 Remote Training & Deployment Guide - DevOps Helper LLM

This guide walks you through the complete **remote training and deployment workflow** using JFrogML platform for the DevOps Helper model.

## 🔐 Login & Configure

First, set up your JFrogML CLI:

```bash
# Install JFrogML CLI (if not already installed)
pip install frogml-cli

# Configure JFrogML CLI interactively
frogml config add --interactive
```

📖 **Need help with installation?** → [JFrogML CLI Installation Guide](https://jfrog.com/help/r/jfrog-ml-documentation/install-jfrog-ml)

---

## 📁 JFrogML Project Structure

```
finetuned_devops_helper/
├── main/                           # Core JFrogML model code
│   ├── model.py                   # LLMFineTuner class with build() and predict() methods
│   ├── config.py                  # Training configuration and hyperparameters
│   ├── data_utils.py              # Dataset loading and preprocessing utilities
│   ├── model_utils.py             # Model loading and hardware optimization
│   ├── conda.yaml                 # Environment dependencies
│   └── __init__.py                # Python package marker
├── test_model_code_locally.py     # Local testing script
└── test_live_endpoint.py          # Endpoint testing script
```

### 📄 File Explanations

- **`model.py`**: Contains `LLMFineTuner` class with `build()` for LLM fine-tuning and `predict()` for inference
- **`config.py`**: Defines model ID, LoRA parameters, training hyperparameters, and prompt formatting
- **`data_utils.py`**: Handles dataset loading, tokenization, and preprocessing for DevOps Q&A
- **`model_utils.py`**: Manages model loading, quantization, and hardware optimization (GPU/CPU/MPS)
- **`conda.yaml`**: Specifies Python environment and dependencies for JFrogML runtime

---

## 🧪 Step 1: Local Testing

Before triggering a build, validate your code locally for faster feedback:

```bash
# Navigate to your project directory
cd finetuned_devops_helper

# Test your model code locally
python test_model_code_locally.py
```

This uses JFrogML's `run_local` utility SDK to validate the code locally before triggering a build.

---

## 🎯 Step 2: Create Model in JFrog UI

1. **Login** to your JFrogML platform
2. **Navigate** to Models section
3. **Click** "Create Model"
4. **Enter** Model ID: `devops_helper_model`
5. **Select** your configuration preferences

---

## 🏗️ Step 3: Build (Training + Packaging)

Run this from your current DevOps helper project directory:

```bash
# Trigger remote training job and model packaging
frogml models build --model-id devops_helper_model . --instance gpu.t4.xl

# The . tells the CLI to pick up code from the current working directory
# Returns a UUID Build ID for deployment
```

**⚡ What happens during build:**
- JFrogML executes the `build()` method in your `LLMFineTuner` class
- Fine-tunes Llama2 8B model using LoRA adapters on DevOps dataset
- Packages the fine-tuned model with all dependencies
- Stores artifacts in JFrog ML Registry

**📋 Build options:**
```bash
frogml models build --help  # See all available parameters
```

---

## 🚀 Step 4: Deploy

### Real-time API Endpoint

Deploy your DevOps helper as a **real-time API** for instant responses:

```bash
# Deploy using the Build ID from step 3
frogml models deploy realtime \
  --model-id devops_helper_model \
  --build-id <your-build-id> \
  --instance gpu.t4.xl \
  --server-workers 1 \
  --replicas 1

# Example with actual UUID:
# frogml models deploy realtime \
#   --model-id devops_helper_model \
#   --build-id a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
#   --instance gpu.t4.xl \
#   --server-workers 1 \
#   --replicas 1
```

**📋 Deployment options:**
```bash
frogml models deploy realtime --help  # See all available parameters
```

### Test Your Deployed Model

```bash
# Test the real-time endpoint
python test_live_endpoint.py
```

---

**✅ Your DevOps Helper model is now deployed and ready for production use.**
