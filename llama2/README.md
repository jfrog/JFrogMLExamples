# LLM with JFrog ML

## Overview

This example is using an LLM which can be pushed to JFrog ML and executed for prompt testing using either the Prompt in JFrog ML or using the [Streamlit RAG app](./streamlit_rag_demo/). For this example we will using the LLM from [Hugging Face "meta-llama/Llama-2-7b-chat-hf"](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

### Features

- **Secure Storage**: Protect your proprietary information by deploying models and additional resources to Artifactory local repositories, giving you fine-grain control of the access to your models.

- **Easy Collaboration**: Share and manage your machine learning projects with your team efficiently.

- **Easy Version Control**: The Machine Learning Repositories SDK (FrogML) provides a user-friendly system to track changes to your projects. You can name, categorize (using namespaces), and keep track of different versions of your work.

## How to Test Locally

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Make sure that you have your evironment set up**: It should be usign Python 3.10.17

3. **Install and Configure the Qwak SDK**: Use your account [JFrog ML API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk)to set up your SDK locally.

    ```bash
    pip install qwak-sdk
    qwak configure --url https://jfrogmldemo.jfrog.io/ --type jfrog  --token <JFrog Acess Token>
    pip install frogml
    ```

4. **Run the Model Locally**: Execute the following command to test the model locally:

    ```bash
   qwak models build . --model-id llm_blm --main-dir main --memory "60GB"  --gpu-compatible
   ```
