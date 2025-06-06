# Perform Sentiment Analysis with BERT and JFrog ML

## Overview

This project demonstrates how to label sentiment in a text prompt using a pre-trained BERT model with [JFrog ML](https://docs.qwak.com/docs/introduction). 

It showcases how to:
- Define the QwakModel class
- Initialize the pre-trained BERT model
- Predict phrase sentiment using JFrog ML's API

The code is designed to work seamlessly with JFrog ML and serves as a practical example.
<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `pyproject.toml` file.

    ```bash
    cd main
    poetry env activate
    poetry -C main install
    ```

3. **Install and Configure the JFrog ML SDK**: Use your account [JFrog ML API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk) to set up your SDK locally.

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

1. **Build on JFrog ML**:

    Create a new model on Qwak using the command:

    ```bash
    qwak models create "BERT Sentiment Analysis" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> ./bert_conda
    ```


2. **Deploy the Model on JFrog ML with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the Qwak Inference SDK:

    ```bash
    pip install qwak-inference
    ```

    Call the Real-Time endpoint using your Model ID from JFrog ML:

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
│   └── conda.yaml         # Conda environment configurationdata
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>