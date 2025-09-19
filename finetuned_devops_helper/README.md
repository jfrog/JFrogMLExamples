# 🛠️ DevOps Helper with Fine-tuned LLM

A **fine-tuned Large Language Model** for DevOps question answering, built with **JFrogML** platform integration.

## 🚀 Quick Start

Choose your preferred workflow:

### **Option 1: Local Model Experimentation & Registry**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   💻 Local      │ -> │   📦 Model      │ -> │   🏗️ Build &    │
│   Training      │    │   Logging to    │    │   Deploy in     │
│   (Notebook)    │    │   JFrog ML      │    │   JFrogML UI    │
│                 │    │   Registry      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**🎯 Recommended for**: Local experimentation and model version logging

**Complete workflow**: [📓 DevOps Helper Training Notebook Guide](devops-helper-training.md)

### **Option 2: ML App Code → Build → Deploy**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   💻 ML App     │ -> │   🏗️ Build      │ -> │   🚀 Deploy     │
│   Code          │    │   (w/ Training  │    │   Real-time API │
│   (main/)       │    │   Job)          │    │   Endpoints     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**🎯 Recommended for**: Standardized, production-ready workflows

**Complete workflow**: [🚀 Remote Training & Deployment Guide](remote-training-and-deployment.md)

---

## 📋 Prerequisites

- **Python**: 3.9-3.11
- **JFrog Account**: [Sign up free](https://jfrog.com/start-free/)
- **Hugging Face Account**: For LLM access

## 📁 Project Structure

```
finetuned_devops_helper/
├── main/                           # Core JFrogML model code
│   ├── __init__.py                # Python package marker
│   ├── conda.yaml                 # Environment dependencies
│   ├── model.py                   # LLMFineTuner class with build() and predict()
│   ├── config.py                  # Model configuration and hyperparameters
│   ├── data_utils.py              # Dataset loading and preprocessing utilities
│   └── model_utils.py             # Model loading and hardware optimization utilities
├── test_model_code_locally.py     # Local model testing script
├── test_live_endpoint.py          # Real-time endpoint testing script
└── README.md                      # This file
```

## 🎯 Model Overview

**Technology**: Fine-tuned **Llama2 8B** model using **LoRA** (Low-Rank Adaptation)
**Domain**: DevOps question answering and assistance
**Method**: Parameter-efficient fine-tuning for domain-specific responses

## 📚 Related Resources

- [JFrogML Documentation](https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction)
- [Model Deployment Strategies](https://jfrog.com/help/r/jfrog-ml-documentation/deploy-models)
- [JFrog Artifactory ML Repositories](https://jfrog.com/help/r/jfrog-artifactory-documentation/machine-learning-repositories)

## 🚀 Next Steps

1. **Choose your workflow** above based on your use case
2. **Follow the linked guides** for step-by-step instructions
3. **Deploy and serve** your fine-tuned DevOps assistant
