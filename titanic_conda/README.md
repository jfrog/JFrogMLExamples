# Titanic Survival Prediction Model with JFrog ML

## Overview

This project employs the CatBoost algorithm for predicting the survival of Titanic passengers. It's implemented using the [JFrog ML](https://docs.qwak.com/docs/introduction) and the CatBoost library.

### Features

- **Custom TitanicSurvivalPrediction Class Definition**: Customizes the base QwakModel to work with the CatBoost algorithm for survival prediction.

- **Model Initialization**: Initializes the CatBoost model with user-defined or default hyperparameters. The model is trained on the Titanic dataset and fine-tuned for optimal performance.

- **Survival Prediction via JFrog ML's Predict API**: Utilizes JFrog ML's Predict API for assessing the probability of survival based on various features like passenger class, age, sex, etc.

### Functionality

The primary functionality is to predict the probability of survival for Titanic passengers. The code is designed for seamless integration with JFrog ML's platform and serves as a practical example for survival prediction tasks.



<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate titanic_conda
    ```

3. **Install and Configure the Qwak SDK**: Use your account [JFrog ML API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk)to set up your SDK locally.

    ```bash
    pip install qwak-sdk
    qwak configure
    ```

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```

<br>

<br>

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML Platform**:

    Create a new model on JFrog ML using the command:

    ```bash
    qwak models create "Titanic Survival Model" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> ./titanic_conda
    ```


2. **Deploy the Model on the JFrog ML Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the JFrog ML Inference SDK:

    ```bash
    pip install qwak-inference
    ```

    Call the Real-Time endpoint using your Model ID from the JFrog ML platform:

    ```bash
    python test_live_mode.py <your-qwak-model-id>
    ```

<br>


## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Titanic Survival Model
│   └── conda.yaml         # Conda environment configuration
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>

