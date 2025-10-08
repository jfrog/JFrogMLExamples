# 📚 Book Recommender System

A content-based book recommendation engine that provides 10 similar book suggestions based on an input ISBN using JFrogML Platform.

## 🚀 Quick Start

### Option 1: [Jupyter Notebook Experimentation](local-training-and-model-registry.ipynb)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Train Model   │───▶│  Log to Model   │───▶│   Package into  │───▶│ Deploy ML API   │
│   in Notebook   │    │    Registry     │    │ Container Image │    │   Endpoint      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
   Jupyter + Python      FrogML Python SDK          JFrogML UI             JFrogML UI
```

### Option 2: [Production ML Pipeline](remote-training-and-deployment.md)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ML App Code    │───▶│ Build & Train   │───▶│ Deploy ML API   │
│   Local IDE     │    │   in Cloud      │    │   Endpoint      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    Local IDE              FrogML CLI             FrogML CLI
```

## 💡 When to Use Each Approach

- **Option 1 (Jupyter Notebook)**: Perfect for data scientists exploring algorithms, testing different features, and rapid prototyping. Train locally, log to JFrogML Registry, then deploy when ready.
- **Option 2 (Production Pipeline)**: Ideal for ML engineers building production systems. Code runs in JFrogML's cloud infrastructure with automatic scaling, monitoring, and CI/CD integration.

## 📁 Project Structure

```
book_recommender/
├── main/                                    # FrogMLModel workflow (production)
│   ├── __init__.py                          # Python package initialization
│   ├── model.py                             # FrogMLModel with build() and predict()
│   ├── data_processor.py                    # Book data preprocessing utilities
│   ├── books_dataset.csv                   # Sample book dataset for training
│   └── conda.yml                            # Conda environment configuration
├── serving_code/                            # Notebook workflow serving code
│   ├── __init__.py                          # Package marker
│   └── predict.py                         # Production predict() function
├── local-training-and-model-registry.ipynb # Notebook workflow (experimentation)
├── test_model_code_locally.py               # Local FrogMLModel testing
├── test_live_endpoint.py                    # Live API endpoint testing
└── remote-training-and-deployment.md       # Remote deployment guide
```

## 📋 Prerequisites

- **Python**: 3.9-3.11
- **JFrog Account**: [Sign up free](https://jfrog.com/start-free/)
- **JFrogML Setup**: [Installation & Configuration Guide](https://jfrog.com/help/r/jfrog-ml-documentation/install-jfrog-ml)
