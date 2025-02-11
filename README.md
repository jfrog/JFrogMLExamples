# JFrog ML Examples

Example projects that demonstrate how to build, train, and deploy ML features and models using the [JFrog ML](https://docs.qwak.com/docs/introduction) product from [JFrog](https://jfrog.com/).

## Table of Contents

1. [Overview](#overview)
2. [Documentation](#documentation)
3. [Getting Started](#getting-started)
4. [Pre-requisites](#pre-requisites)
5. [Developer Environment Example](#development-environment-example)
6. [JFrog Model Examples](#jfrog-model-examples)

## Overview

This repository contains example projects that showcase the capabilities of the JFrog ML for MLOps. Each project is designed to be a standalone example, demonstrating different aspects of machine learning, from data preprocessing to model building and deployment.

## Documentation

All documentation for the JFrog ML Platform can be found on the [JFrog ML Documentation](https://docs.qwak.com/docs/introduction) website.

## Getting Started

To get started with these examples:

1. Clone this repository.
2. Navigate to the example project you're interested in.
3. Follow the README and installation instructions within each project folder.

## Pre-requisites

To use the JFrog ML Platform for MLOps, you will need:

1. A JFrog Platform account with access to the JFrog ML Platform.
2. A termial or command line interface with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Python](https://www.python.org/downloads/) installed. JFrog ML supports < 3.10 versions of Python.
3. A [Poetry](https://python-poetry.org/docs/) installation for managing Python dependencies.

## Development Environment Example 
We have provided a sample setup for developers to use JFrog ML more effectively. 
[Developer Environment Example](./environment_example/README.md)

## JFrog Model Examples
| Example | Category | Model | Info |
|---------|------|----------|------|
| [Customer Churn Analysis](./churn_model_new/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![XGBoost](https://img.shields.io/badge/-XGBoost-%23D3D3D3) | Predicts Telecom subscriber churn using XGBoost [Conda]. |
| [Credit Risk Assesment](./catboost_poetry/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | Predicts loan default risk using CatBoost algorithm [Poetry] |
| [Sentiment Analysis](./bert_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | Performs binary sentiment analysis using a pre-trained BERT model. |
| [Titanic Survival Prediction](./titanic_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | Binary classification model for Titanic survival prediction.[Conda] |
| [Sentiment Analysis](./sentiment_analysis/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![Pandas](https://img.shields.io/badge/-Transformers-%23D3D3D3) | Sentiment Analysis Model with JFrog ML [Poetry] |
| [FrogML Example](./frogMLExample/) | ![JFrogML](https://img.shields.io/pypi/pyversions/:frogml) | ![FrogML]((https://img.shields.io/pypi/pyversions/:frogml) | Example of JFrog ML Repo and FrogML [JFrog] |


