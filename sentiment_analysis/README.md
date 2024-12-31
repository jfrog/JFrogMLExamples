# Sentiment Analysis Model with JFrog ML

## Overview

This project employs the Sentiment Analysis. It's implemented using the [JFrog ML](https://docs.qwak.com/docs/introduction).

### Features

<br>

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML Platform**:

    Create a new model on JFrog ML using the command:

    ```bash
    qwak models create "Sentiment Analysis" --project "Sample Project"
    ```

    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> ./main
    ```

2. **Deploy the Model on the JFrog ML Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

<br>

## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── finetuning.py      # Fine-tuning script
│   ├── model.py           # Defines the Sentiment Analysis Model
│   └── poetry.yaml        # Poetry configuration file
|   └── pyproject.toml    
└── README.md              # Documentation
```
<br>
