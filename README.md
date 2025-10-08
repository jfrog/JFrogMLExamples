# 🚀 JFrog ML Examples

Collection of **machine learning examples** demonstrating how to build, train, and deploy ML models using the **[JFrogML Platform](https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction)**.

## 🚀 Getting Started

To get started with these examples:

1. **Clone** this repository
2. **Navigate** to the example project you're interested in
3. **Follow** the README and installation instructions within each project folder

## 📋 Prerequisites

- **Python**: 3.9-3.11
- **JFrog Account**: [Sign up free](https://jfrog.com/start-free/)
- **JFrogML Setup**: [Installation & Configuration Guide](https://jfrog.com/help/r/jfrog-ml-documentation/install-jfrog-ml)

<br>

## 🤖 ML Examples

Click any example below to open a step-by-step guide for building, training, and deploying it.

| **Example** | **Domain** | **Technology** | **Description** |
|-------------|------------|----------------|-----------------|
| **[💳 Fraud Detection](./fraud_detection/)** | Financial | **CatBoost + XGBoost + RF** | Credit card fraud detection with ensemble methods |
| **[🛠️ DevOps Helper](./finetuned_devops_helper/)** | DevOps | **Fine-tuned Llama/Qwen LLM** | DevOps assistant using fine-tuned Llama2 8B and Qwen 1.5B with LoRA |
| **[📚 Book Recommender](./book_recommender/)** | E-commerce | **Content-Based Filtering** | ISBN-based book recommendation system using TF-IDF and cosine similarity |
| **[🏪 Feature Store Quickstart](./feature_store_quickstart_guide/)** | Feature Engineering | **Spark SQL + Feature Store** | Complete guide to JFrogML Feature Store |
| **[💰 Financial QA](./finetuned_financial_qa/)** | FinTech | **Fine-tuned T5** | Question answering for financial domain using T5 with LoRA |
| **[📞 Customer Churn](./churn_model/)** | Telecom | **XGBoost** | Subscriber churn prediction with gradient boosting |

<br>

## 🔄 Two Ways to Deploy

Pick the workflow that fits your team. Both are production-ready; they differ in how you control builds and versioning.

### 🔬 Artifact-first (Registry)
- Train in a notebook/script and log a framework-native model binary to the JFrogML Registry
- The logged model version includes dependency manifest, serving code, and metadata
- JFrogML packages it into a container image; you deploy the image as realtime/batch/streaming API


### 🚀 Code-first (FrogMLModel)
- Implement the lifecycle in code (train/initialize/serve) with a `FrogMLModel` in your repo
- Trigger a Build; JFrogML builds your code, runs training if defined or preloads a binary
- You deploy the Build as realtime/batch/streaming API


#### Details at a glance

| **Aspect** | **🔬 Artifact-first (Registry)** | **🚀 Code-first (FrogMLModel)** |
|------------|----------------------------------|---------------------------------|
| **Authoring** | Train in notebook/script; produce a model binary | Develop in repo; wrap logic in `FrogMLModel` |
| **What is logged/pushed** | Binary model artifact to JFrogML Registry (framework-native: scikit-learn, PyTorch, ONNX, etc.) + dependency manifest, serving code, metadata | Source code pushed/triggered for build (`FrogMLModel` + repo code); no binary logged at this step |
| **Versioning** | Versioned ML native artifacts in JFrogML Registry | Versioned Builds in JFrogML
| **Build semantics** | Packaging the logged binary into a container image | Build executes your custom workflow; may run training or preload a binary |
| **Deployment** | Deploy as API (realtime/batch/streaming) from the built image (same after build) | Deploy as API (realtime/batch/streaming) from the built image (same after build) |
| **Who drives workflow** | Artifact + metadata; platform packages and serves | Your code defines build/train/serve lifecycle |
| **Production posture** | Production-capable; simpler path with less custom control | Production-capable; greater control and standardization |