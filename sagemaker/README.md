# SageMaker + JFrog Artifactory

Train in SageMaker, store model artifacts in JFrog Artifactory with `frogml`, then serve them locally and on a SageMaker endpoint.

## Prerequisites

- Python 3.9-3.11
- AWS credentials configured
- JFrog account and `frogml` installed
- AWS Secrets Manager entries for:
  - `HF_TOKEN_SECRET_ID`
  - `JF_ACCESS_TOKEN_SECRET_ID`

## Quick start

1. Configure the CLI:

   `frogml config add --interactive`

2. Open and run `pipeline.ipynb` top to bottom.
3. After training, set `MODEL_VERSION` in the notebook to the version logged to Artifactory.

## Project structure

```
sagemaker/
├── pipeline.ipynb        # End-to-end training + inference demo
├── training/             # Training code and hyperparameters
├── deployment/           # Inference code for SageMaker
└── pyproject.toml
```

## Related resources

- JFrogML documentation: https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction
