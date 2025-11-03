# Generates Financial Text using Fine-Tuned FLAN-T5

## Overview

This project leverages a fine-tuned FLAN-T5 model for generating text based on financial questions and answers. It's implemented using [JFrog ML's Machine Learning Platform](https://jfrog.com/jfrog-ml/) and the T5 library.

### Features

- **Custom FLAN-T5 Class Definition**: Customizes the base BaseModel to work with the fine-tuned FLAN-T5 model for financial text generation.
  
- **Model Initialization**: Initializes the FLAN-T5 base model with user-defined hyperparameters. The model is trained on a financial Q&A dataset and fine-tuned for optimal text generation.

- **Financial Text Generation via JFrog ML's Predict API**: Utilizes JFrog ML's Predict API for generating text based on financial prompts.

### Functionality

The primary functionality is to generate text for financial questions and answers. The code is designed for seamless integration with JFrog ML's platform and serves as a practical example for financial text generation tasks.


<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `pyproject.toml` file.

    ```bash
    poetry -C main install
    ```

3. **[Install and Configure the frogml](https://jfrog.com/help/r/jfrog-ml-documentation/setting-up-jfrog-ml)**.

    ```bash
    pip install forgml-cli
    frogml configure
    ```

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   poetry run python test_model_locally.py
   ```

<br>

<br>

## How to Run Remotely on JFrog Ml

1. **Build on the JFrog ML Platform**:

    Create a new model on JFrog ML using the command:

    ```bash
    frogml models create "Finetuned Flan T5" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    frogml models build --model-id <your-model-id> .
    ```


2. **Deploy the Model on the JFrog ML Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    frgoml models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the JFrog ML Inference SDK:

    ```bash
    pip install frogml-inference
    ```

    Call the Real-Time endpoint using your Model ID from the JFrogML platform:

    ```bash
    python test_live_mode.py <your-jfrogml-model-id>
    ```

<br>


## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Flan T5 Base Model
│   ├── training.py        # Trains the model on the financial dataset Model
│   ├── helpers.py         # Defines a variety of helpers for the Model
│   ├── dataset_loader.py  # Loads the online dataset
│   └── pyproject.toml     # Poetry configuration
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/jfrog-ml/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrogML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/jfrog-ml/)
