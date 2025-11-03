# DistilGPT2 Text Generation Model with Transformers and jFrog ML

## Overview

This project uses a simplified version of the GPT-2 model, known as DistilGPT2, for efficient text generation. It's implemented using the [JFrog ML's Machine Learning Platform](https://jfrog.com/jfrog-ml/)  and Transformers libraries.

It covers:
- BaseModel class definition
- Model initialization
- Text generation via JFrog ML's Predict API

The code is designed to work seamlessly with JFrog ML's platform and serves as a practical example.

<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate distilgpt2
    ```

3. **[Install and Configure the frogml SDK](https://jfrog.com/help/r/jfrog-ml-documentation/setting-up-jfrog-ml)**.

    ```bash
    pip install forgml-cli
    frogml configure
    ```

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```

<br>

<br>

## How to Run Remotely on JFrog ML

1. **Build on the JFrgo ML Platform**:

    Create a new model on JFrog ML using the command:

    ```bash
    forgml models create "DistilGPT2 LLM" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    frogml models build --model-id <your-model-id> ./distilgpt2_conda
    ```


2. **Deploy the Model on the JFrog ML Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    frogml models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the JFrog ML Inference SDK:

    ```bash
    pip install frogml-inference
    ```

    Call the Real-Time endpoint using your Model ID from the JFrog ML platform:

    ```bash
    python test_live_mode.py <your-jfrogml-model-id>
    ```

<br>


## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Code Generation Model
│   └── conda.yaml         # Conda environment configurationdata
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog](https://jfrog.com/jfrog-ml/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/jfrog-ml/)
