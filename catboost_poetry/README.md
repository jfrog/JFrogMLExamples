# Predicts loan default risk using CatBoost

## Overview

This project leverages the CatBoost machine learning algorithm for credit risk assessment. It's implemented using [JFrog ML](https://docs.qwak.com/docs/introduction) and the CatBoost library.

### Features

- **Custom CatBoost Class Definition**: Customizes the base QwakModel to work with the CatBoost algorithm for credit risk prediction.
  
- **Model Initialization**: Initializes the CatBoost model with user-defined or default hyperparameters. The model is trained on a credit risk dataset and fine-tuned for optimal performance.

- **Credit Risk Prediction via JFrog ML's Predict API**: Utilizes JFrog's Predict API for assessing the probability of default based on various features like age, sex, job, housing, etc.

### Functionality

The primary functionality is to predict the probability of default for credit applications. The code is designed for seamless integration with JFrog ML and serves as a practical example for credit risk assessment tasks.


<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f ./main/conda.yml
    conda activate catboost_poetry
    ```

3. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `pyproject.toml` file.

    ```bash
    poetry -C main install
    ```

4. **Install and Configure the JFrog Ml SDK**: Use your account [JFrog ML API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk) to set up your SDK locally.

    ```bash
    pip install qwak-sdk
    qwak configure
    ```

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   poetry run python test_model_locally.py
   ```

<br>

<br>

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML**:

    Create a new model on Qwak using the command:

    ```bash
    qwak models create "Credit Risk" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> .
    ```


2. **Deploy the Model on the JFrog ML with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the Qwak Inference SDK:

    ```bash
    pip install qwak-inference
    ```

    Call the Real-Time endpoint using your Model ID from the JFrog ML:

    ```bash
    python test_live_mode.py <your-qwak-model-id>
    ```

<br>


## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Credit Risk Model
│   ├── data.csv           # Defines the data to train the Model
│   └── pyproject.toml     # Poetry configuration
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>