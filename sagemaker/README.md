# 🤖 SageMaker + JFrog Artifactory

## 🎯 Overview

Fine-tune **Qwen 1.5 (0.5B)** on a DevOps QA dataset using LoRA in **AWS SageMaker**, register the model in **JFrogML**, then serve it locally and on a SageMaker endpoint.

Unlike the other examples in this repository (which use the code-first `FrogMLModel` pattern), this example follows an **artifact-first** approach: train on an external platform, register versioned model artifacts in JFrogML, and load them at serving time.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  🔗 Curate       │ -> │  🏋️ Fine-tune    │ -> │  📦 Register     │ -> │  🚀 Deploy       │
│  Base model via  │    │  LoRA training   │    │  Model version   │    │  Load from       │
│  JFrog Remote    │    │  in SageMaker    │    │  in JFrogML      │    │  JFrogML & serve │
│  Repository      │    │                  │    │  (Artifactory)   │    │  on SageMaker    │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
     JFrog HF Remote        AWS SageMaker         frogml Python SDK       frogml Python SDK
```

**How JFrog fits in:**

- **Curation** — A JFrog remote repository proxies Hugging Face, giving you caching, access control, and visibility over base model downloads
- **Registration** — `frogml.huggingface.log_model` publishes the fine-tuned model as a versioned artifact in Artifactory with full metadata (hyperparameters, metrics, dataset lineage)
- **Deployment** — `frogml.huggingface.load_model` retrieves the exact model version at inference time, ensuring reproducibility

<br>

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed
- **AWS account** with:
  - An IAM role with SageMaker and Secrets Manager permissions
  - AWS credentials configured locally (`aws configure`)
  - Secrets stored in AWS Secrets Manager:
    - `jfrog/hf_token` — Hugging Face token (for model downloads through JFrog)
    - `jfrog/jf_token` — JFrog access token (for artifact uploads)
- **JFrog account** ([Get started for free](https://jfrog.com/start-free/)) with:
  - An Artifactory Machine Learning type repository
  - A Hugging Face remote repository configured
  - `frogml` installed (`pip install frogml`)

<br>

## 🚀 Quick Start

1. Install dependencies:

   ```bash
   pip install frogml sagemaker boto3
   ```

2. Configure `frogml`:

   ```bash
   frogml config add --interactive
   ```

3. Open [`pipeline.ipynb`](pipeline.ipynb), fill in the configuration cell at the top, and run cells in order.

<br>

## 🔬 How It Works

| Stage | What happens | Key code |
|-------|-------------|----------|
| **Curate** | Base model is downloaded through a JFrog remote repository that proxies Hugging Face | `snapshot_download()` with `HF_ENDPOINT` pointing to JFrog |
| **Fine-tune** | SageMaker training job applies LoRA (rank 16) to the base model on the `Szaid3680/Devops` dataset | `training/train.py` via `ModelTrainer` |
| **Register** | Fine-tuned model + tokenizer are logged as a versioned artifact in JFrogML with hyperparameters and metrics | `frogml.huggingface.log_model()` |
| **Deploy** | Model version is loaded from JFrogML and served via a HuggingFace text-generation pipeline | `frogml.huggingface.load_model()` in `deployment/inference.py` |

<br>

## 📓 Notebook Walkthrough

The [`pipeline.ipynb`](pipeline.ipynb) notebook is organized into these sections:

1. **Setup** — Configure JFrog and AWS settings (single cell to fill in)
2. **Training** — Launch the SageMaker training job; model is registered in JFrogML automatically
3. **Inference** — Load the model from JFrogML, build and test locally (in-process, no containers)
4. **SageMaker Endpoint** — Package as a SageMaker model, deploy to a managed endpoint, test
5. **Cleanup** — Delete the endpoint to avoid charges

<br>

## ⚙️ Configuration Reference

### Environment Variables

All settings are consolidated in the first code cell of the notebook:

| Variable | Description |
|----------|-------------|
| `JF_URL` | Your JFrog platform URL (e.g. `https://company.jfrog.io`) |
| `JF_REPO` | Artifactory Machine Learning type repository name |
| `JF_PROJECT` | Artifactory project key |
| `AWS_ROLE` | IAM role ARN for SageMaker execution |
| `HF_ENDPOINT` | JFrog Hugging Face remote repository URL |
| `HF_TOKEN_SECRET_ID` | Secrets Manager key for the Hugging Face token |
| `JF_ACCESS_TOKEN_SECRET_ID` | Secrets Manager key for the JFrog access token |

### Hyperparameters

Defined in [`training/hyperparameters.json`](training/hyperparameters.json):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 1 | Number of training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `max_steps` | 1 | Max training steps (overrides epochs; increase for real training) |
| `gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `train_batch_size` | 1 | Per-device training batch size |
| `dataset_sample_percentage` | 0.1 | Percentage of dataset to use (increase for real training) |

> The defaults are intentionally small for a quick smoke test. For meaningful fine-tuning, set `max_steps` to `-1` (use epochs) and `dataset_sample_percentage` to `100`.

### Model & LoRA Settings

Defined in [`training/config.py`](training/config.py):

| Setting | Value | Description |
|---------|-------|-------------|
| `MODEL_ID` | `Qwen/Qwen1.5-0.5B-Chat` | Base model from Hugging Face |
| `LORA_CONFIG.r` | 16 | LoRA rank |
| `LORA_CONFIG.lora_alpha` | 32 | LoRA scaling factor |
| `LORA_CONFIG.target_modules` | `q_proj, k_proj, v_proj, o_proj` | Attention layers to adapt |
| `MAX_SEQ_LENGTH` | 256 | Maximum sequence length for training |

<br>

## 📁 Project Structure

```
sagemaker/
├── README.md
├── pipeline.ipynb              # End-to-end notebook (start here)
├── pyproject.toml              # Package config for SageMaker inference
├── training/                   # SageMaker training job source bundle
│   ├── train.py                # Training entry point
│   ├── config.py               # Model, LoRA, and quantization settings
│   ├── dataset_utils.py        # Dataset loading and formatting
│   ├── hyperparameters.json    # Tunable training parameters
│   └── requirements.txt
└── deployment/                 # SageMaker inference source bundle
    ├── __init__.py
    ├── inference.py            # InferenceSpec — loads model from JFrogML
    ├── config.py               # Model name and repo settings
    └── requirements.txt
```

> `training/` and `deployment/` are packaged as **independent SageMaker source bundles**, which is why some configuration values (model name, repo, secrets helper) are duplicated between them.

<br>

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValueError: HF_TOKEN_SECRET_ID is not set` | Ensure the environment variables are passed to the SageMaker training job via the `env` dict in the notebook |
| `ValueError: JF_ACCESS_TOKEN_SECRET_ID is not set` | Same as above — check the notebook `env` dict and Secrets Manager |
| `No CUDA detected` message during training | Expected on CPU instances; training will work but slower. Use a GPU instance (`ml.g4dn.xlarge` or larger) for production training |
| Model version not found at inference time | Check that `MODEL_VERSION` matches the version string from the training job logs |
| `AccessDeniedException` from Secrets Manager | Ensure the SageMaker execution role has `secretsmanager:GetSecretValue` permission |

<br>

## 🔗 Related Resources

- [JFrogML Documentation](https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction)
- [JFrog Hugging Face Integration](https://jfrog.com/help/r/jfrog-artifactory-documentation/hugging-face-repositories)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Other JFrogML Examples](../README.md)

## 🤝 Contributing

Found an issue or have a suggestion? Please:
1. Check existing [issues](../../issues)
2. Review the relevant guide
3. Submit a pull request with improvements

## 📚 Next Steps

1. **Run the notebook** — Follow `pipeline.ipynb` end-to-end
2. **Tune hyperparameters** — Increase `max_steps` and `dataset_sample_percentage` in `hyperparameters.json` for better results
3. **Swap the model** — Change `MODEL_ID` in `training/config.py` to fine-tune a different base model
4. **Explore other examples** — See the [repository root](../README.md) for code-first `FrogMLModel` examples
