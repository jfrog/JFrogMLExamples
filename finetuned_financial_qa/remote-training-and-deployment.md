# 🚀 Remote Training & Deployment (JFrogML)

## 🎯 Overview

This guide covers how to build and deploy the Financial QA model on JFrogML. The **training happens during the build process** via the `build()` method in your FrogMLModel, and the `predict()` method handles serving.

---

## 🔐 Step 1: Login & Configure

- Install CLI:
```bash
pip install frogml-cli
```

- Login and configure credentials (interactive):
```bash
frogml config add --interactive
```
Refer to the JFrog ML install and setup instructions: [Install JFrog ML](https://jfrog.com/help/r/jfrog-ml-documentation/install-jfrog-ml).

---

## 📁 Step 2: JFrogML Project Structure

Understanding the required project structure for JFrogML deployment:

```
.
├── main/                       # Main directory containing core code
│   ├── __init__.py             # Python package marker (required)
│   ├── model.py                # FrogMLModel implementation with build() and predict()
│   ├── training.py             # Model training utilities for T5 fine-tuning
│   ├── helpers.py              # Helper functions for data loading and device setup
│   ├── dataset_loader.py       # Financial dataset loading utilities
│   └── pyproject.toml          # Poetry dependencies and environment
```

### **File Explanations**

- **`main/`**: Directory containing all core model code and dependencies
- **`__init__.py`**: Empty file that makes `main/` a Python package for imports
- **`model.py`**: FrogMLModel class with key methods:
  - `build()`: Fine-tuning logic for FLAN-T5 (runs during build process)
  - `initialize_model()`: Runtime initialization at deployment (loads tokenizer)
  - `predict()`: Text generation logic (runs during serving)
  - `schema()`: Input validation for financial prompts
- **`training.py`**: T5 model fine-tuning utilities and training loop
- **`helpers.py`**: Data loading and device detection utilities
- **`dataset_loader.py`**: Financial QA dataset loading and preprocessing
- **`pyproject.toml`**: Poetry dependencies (transformers, torch, etc.)

---

## 🧪 Step 3: Local Testing

Before building on JFrogML, validate your code locally for faster feedback:

```bash
# Test your model locally using JFrogML's run_local utility
python test_model_code_locally.py
```

This uses JFrogML's `run_local` SDK utility to:
- Validate your `FrogMLModel` implementation
- Test `build()` and `predict()` methods locally
- Catch issues before triggering remote builds
- Provide faster development iteration

---

## 🎯 Step 4: Create Model in JFrog UI

Before building, create your model in the JFrog platform:

1. **Navigate to JFrog UI** → **AI/ML** section
2. **Create New Model** → Name: "Financial QA with Fine-tuned FLAN-T5" 
3. **Copy the Model ID** generated (you'll need this for CLI commands)

This associates your code with a specific model in the JFrog platform for tracking and management.

---

## 🏗️ Step 5: Build (Training + Packaging)

The build process executes your `build()` method (which contains T5 fine-tuning logic) and packages everything for deployment:

```bash
# Build the model (run from finetuned_financial_qa/ directory - the . picks up code from current dir)
frogml models build --model-id financial_qa_model . --instance medium

# This will return a Build ID (UUID) - copy it for deployment
# Example output: Build ID: f47ac10b-58cc-4372-a567-0e02b2c3d479

# View build logs (includes training logs)
frogml models builds logs -b <your_build_id> -f 

# See all build command parameters
frogml models build --help
```

**What happens during build:**
1. **Environment Setup**: Installs dependencies from `pyproject.toml`
2. **Fine-tuning**: Your `build()` method runs T5 fine-tuning on financial QA dataset
3. **Model Packaging**: Creates deployment-ready container with fine-tuned model
4. **Validation**: Ensures model and serving logic are ready for inference
5. **Build ID Generated**: Copy this UUID for deployment commands

---

## 🚀 Step 6: Deploy

### Real-time API
```bash
# Deploy as real-time endpoint (use the Build ID from previous step)
frogml models deploy realtime --model-id financial_qa_model --build-id <your-build-id>

# See all realtime deployment parameters
frogml models deploy realtime --help
```

Test the endpoint:
```bash
python test_live_endpoint.py
```

### Batch Processing
```bash
# Deploy for batch inference (use the Build ID from previous step)
frogml models deploy batch --model-id financial_qa_model --build-id <your-build-id>

# See all batch deployment parameters
frogml models deploy batch --help
```

Submit a batch job (example):
```bash
python test_batch_endpoint.py
```

<br>

## 🎯 JFrogML Platform Benefits

### **Integrated ML Lifecycle**
- **Code to Production**: Single platform for building, training, and serving ML models
- **FrogMLModel Framework**: Standardized approach with `build()` for training and `predict()` for serving
- **Automatic Scaling**: Handle varying inference loads automatically

### **T5 Model Optimization**
- **GPU Acceleration**: Automatic GPU allocation for T5 fine-tuning during build
- **Model Serving**: Optimized inference runtime for transformer models
- **Memory Management**: Efficient handling of large language models

### **Production Features**
- **Model Versioning**: Track different fine-tuned model versions
- **A/B Testing**: Compare different T5 configurations
- **Monitoring**: Track model performance and inference metrics
- **Security**: Enterprise-grade security for model deployment

---

**🎯 Ready to deploy your Financial QA model?** Follow the steps above to get your fine-tuned FLAN-T5 model running in production! 🚀
