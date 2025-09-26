# Perform Sentiment Analysis with BERT and FrogML

## Overview

This project demonstrates how to label sentiment in a text prompt using a pre-trained BERT model with [FrogML](`https://jfrog.com/help/r/jfrog-ml-documentation/get-started-with-jfrog-ml`). 

It showcases how to:
- Define the FrogMLModel class
- Initialize the pre-trained BERT model
- Predict phrase sentiment using FrogML's API

The code is designed to work seamlessly with FrogML and serves as a practical example.
<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `pyproject.toml` file.

    ```bash
    cd main
    poetry env activate
    poetry -C main install
    ```

3. **Install and Configure the FrogML SDK**: Use your account [FrogML API Key](`https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-quickstart`) to set up your SDK locally.

    ```bash
    pip install frogml frogml-cli
    frogml config add 
    ```

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```

<br>

<br>

## How to Run Remotely on JFrog ML

1. **Build on JFrog ML**:

    Create a new model on FrogML using the command:

    ```bash
    frogml models create "BERT Sentiment Analysis" --project-key "<Project Name>"
    ```


    Initiate a model build with:

    ```bash
    frogml models build .  --model-id bert_sentiment_analysis --main-dir main --memory "60GB"  --gpu-compatible --name bert-test-01
    ```


2. **Deploy the Model on FrogML with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    frogml models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the FrogML Inference SDK:

    ```bash
    pip install frogml-inference
    ```

    Call the Real-Time endpoint using your Model ID from FrogML:

    ```bash
    python test_live_mode.py <your-frogml-model-id>
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