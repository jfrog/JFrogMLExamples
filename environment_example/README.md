# JFrog ML Developer Evnironment Setup Example

## Reasons for using this example environment setup

The current CLI tool for JFrog ML only supports Python versions < 3.10. This example environment setup is designed to help developers use JFrog ML more effectively by creating a Conda environment with the required Python version and dependencies.

## Steps to setup the environment

1. Clone this repository.
2. Install Conda, Poetry, and Python 3.9.
3. A termial or command line interface with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Python](https://www.python.org/downloads/) installed. JFrog ML supports < 3.10 versions of Python.
4. A [Poetry](https://python-poetry.org/docs/) installation for managing Python dependencies.
5. Create a Conda environment with the required Python version and dependencies.

```
conda create --prefix ./env python=3.9
```

6. Activate the Conda environment.

```
conda activate ./env
python -V
```

7. Install the required dependencies using Conda

```
conda install jupyter pandas numpy matplotlib scikit-learn tqdm
```

8. Install Python dependencies for ML

```
pip install torch torchvision torchaudio
```

9. Install the JFrog ML SDK

```
pip install qwak-sdk
```

## Next Steps

Now that you have set up the environment, you can start using the JFrog ML SDK to build, train, and deploy ML models. For more information, refer to the [JFrog ML Documentation](https://docs.qwak.com/docs/introduction).
