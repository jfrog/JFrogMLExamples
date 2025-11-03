# 💰 Financial QA with Fine-tuned FLAN-T5

## 🎯 Overview

This project demonstrates a **Financial Question Answering Model** using fine-tuned FLAN-T5 and the JFrogML platform. It showcases multiple deployment strategies and training approaches for production-ready financial text generation systems.

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.9-3.11** installed
- **JFrog account** ([Get started for free](https://jfrog.com/start-free/))

<br>

## 🚀 Quick Start

Choose your preferred approach:

<br>

### 🏠 **Option 1: Local Model Experimentation & Registry**

```
┌─────────────────────┐    ┌─────────────────────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🔧 Local Model    │ -> │   📦 JFrog ML Registry              │ -> │   🏗️ Build      │ -> │   🚀 Deploy     │
│   Experimentation   │    │   (Artifactory)                     │    │   Container     │    │   ML Serving    │
│   (Training)        │    │                                     │    │   Image         │    │   API Endpoint  │
└─────────────────────┘    └─────────────────────────────────────┘    └─────────────────┘    └─────────────────┘
     FrogML Python SDK              FrogML Python SDK                     JFrogML UI              JFrogML UI
```

**Complete workflow**: [📓 Training Notebook](financial-qa-training.ipynb)

**Best for**: Experimentation, model versioning, and custom serving behavior development

---
<br>

### ☁️ **Option 2: ML App Code → Build → Deploy**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   💻 Local ML   │ -> │   🏗️ Build      │ -> │   🚀 Deploy     │
│   App Code      │    │    Process      │    │   ML Serving    │
│   (or GitHub)   │    │(w/ Training Job)│    │   API Endpoint  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Local IDE              FrogML CLI           FrogML UI/CLI
```

**Complete workflow**: [🚀 Remote Training & Deployment Guide](remote-training-and-deployment.md)

**Best for**: Standardized, replicable, production-ready workflows with integrated training and serving

---

**💡 Recommendation**: 
- **Choose Option 1** if you want to experiment locally and push experiments to model registry (JFrog Artifactory) with all metadata
- **Choose Option 2** if you want a standardized, production-ready workflow with integrated training and serving

<br>


## 📁 Project Structure

```
finetuned_financial_qa/
├── main/                       # Main directory containing core code
│   ├── __init__.py             # Python package initialization
│   ├── model.py                # FrogMLModel with financial QA logic
│   ├── training.py             # Model training utilities
│   ├── helpers.py              # Helper functions for the model
│   ├── dataset_loader.py       # Financial dataset loading utilities
│   └── pyproject.toml          # Poetry dependencies
```

<br>

## 🔗 Related Resources

- [JFrogML Documentation](https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction)
- [Feature Store Example](../feature_set_quickstart_guide/README.md)

## 🤝 Contributing

Found an issue or have a suggestion? Please:
1. Check existing [issues](../../issues)
2. Review the relevant guide
3. Submit a pull request with improvements

## 📚 Next Steps

1. **Choose your deployment path** from the guides above
2. **Follow the step-by-step instructions** in your chosen guide
3. **Customize the model** for your specific financial QA needs
4. **Scale up** with larger datasets and more complex fine-tuning

---

**Ready to get started?** Pick a guide above and begin your financial QA journey! 🚀