# FrogML SDK with JFrog ML Repository

## Overview

Machine Learning Repositories with the FrogML SDK is a local management framework tailored for machine learning projects, serving as a central storage for models and artifacts, featuring a robust version control system. It offers local repositories and an SDK for effortless model deployment and resolution. It's implemented using the [JFrog ML Repository](https://jfrog.com/help/r/jfrog-artifactory-documentation/machine-learning-repositories) and the [FrogML SDK](https://jfrog.com/help/r/jfrog-artifactory-documentation/frogml-library).

### Features

- **Secure Storage**: Protect your proprietary information by deploying models and additional resources to Artifactory local repositories, giving you fine-grain control of the access to your models.

- **Easy Collaboration**: Share and manage your machine learning projects with your team efficiently.

- **Easy Version Control**: The Machine Learning Repositories SDK (FrogML) provides a user-friendly system to track changes to your projects. You can name, categorize (using namespaces), and keep track of different versions of your work.

## How to Test Locally

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yml
    conda activate RiskModelDemo
    ```

3. **Install and Configure the Qwak SDK**: Use your account [JFrog ML API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk)to set up your SDK locally.

    ```bash
    pip install qwak-sdk
    qwak configure --url https://jfrogmldemo.jfrog.io/ --type jfrog  --token <JFrog Acess Token>
    ```

4. **Run the Model Locally**: Execute the following command to test the model locally:

   Open the FrogML Demo.ipynb file in Jupyter Notebook and run the cells.
