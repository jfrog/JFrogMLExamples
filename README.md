# 🚀 JFrog ML Examples

Collection of **machine learning examples** demonstrating how to build, train, and deploy ML models using the **[JFrogML Platform](https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction)**.

## 🎯 Overview

Production-ready ML examples showcasing complete workflows on JFrogML Platform. Each example provides two deployment paths:

**🔬 Local Training → Registry**: Train models locally, publish to JFrogML Model Registry, then build and deploy as API endpoints.  
**🚀 Remote Training → Deploy**: Trigger remote training jobs, build/package models, and deploy as scalable API endpoints.

Choose any example and follow the guided workflow that fits your development needs.

## 🚀 Getting Started

To get started with these examples:

1. **Clone** this repository
2. **Navigate** to the example project you're interested in
3. **Follow** the README and installation instructions within each project folder

## 📋 Prerequisites

- **Python**: 3.9-3.11
- **JFrog Account**: [Sign up free](https://jfrog.com/start-free/)
- **JFrogML Setup**: [Installation & Configuration Guide](https://jfrog.com/help/r/jfrog-ml-documentation/install-jfrog-ml)

## 🤖 ML Examples

| **Example** | **Domain** | **Technology** | **Description** |
|-------------|------------|----------------|-----------------|
| **[💳 Fraud Detection](./fraud_detection/)** | Financial | **CatBoost + XGBoost + RF** | Credit card fraud detection with ensemble methods |
| **[🛠️ DevOps Helper](./finetuned_devops_helper/)** | DevOps | **Fine-tuned Llama/Qwen LLM** | DevOps assistant using fine-tuned Llama2 8B and Qwen 1.5B with LoRA |
| **[🏪 Feature Store Quickstart](./feature_set_quickstart_guide/)** | Feature Engineering | **Spark SQL + Feature Store** | Complete guide to JFrogML Feature Store |
| **[💰 Financial QA](./finetuned_financial_qa/)** | FinTech | **Fine-tuned T5** | Question answering for financial domain using T5 with LoRA |
| **[📞 Customer Churn](./churn_model_new/)** | Telecom | **XGBoost** | Subscriber churn prediction with gradient boosting |
| **[💳 Credit Risk](./catboost_poetry/)** | Finance | **CatBoost** | Loan default risk assessment |
| **[😊 BERT Sentiment](./bert_conda/)** | NLP | **BERT** | Binary sentiment analysis with pre-trained BERT |
| **[📝 Sentiment Analysis](./sentiment_analysis/)** | NLP | **Transformers** | General sentiment classification |
| **[🍷 Wine Classification](./wine-type-training/)** | Food & Beverage | **Deep Learning** | Wine type prediction using neural networks |