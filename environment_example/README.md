# JFrog ML Developer Evnironment Setup Example

## Reasons for using this example environment setup

The current CLI tool for JFrog ML only supports Python versions < 3.10. This example environment setup is designed to help developers use JFrog ML more effectively by creating a Conda environment with the required Python version and dependencies.

## Steps to setup the environment

1. Clone this repository.
2. Install Conda, Poetry, and Python 3.10.17
3. A termial or command line interface with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Python](https://www.python.org/downloads/) installed. JFrog ML supports < 3.10 versions of Python.
4. A [Poetry](https://python-poetry.org/docs/) installation for managing Python dependencies.
5. Install pyenv

First, you need to install pyenv itself. The exact method depends on your operating system, but here's a general approach:

macOS (using Homebrew): brew install pyenv 🍻
Linux (using your distribution's package manager): This varies greatly. Check your distribution's documentation. For example, on Debian/Ubuntu, you might use apt install pyenv. 🐧
Other methods: You can find instructions for other installation methods on the official pyenv GitHub repository. This is usually the most reliable source. 💻

6. Install Python 3.10.17

Now that pyenv is installed, let's get Python 3.10.17:

Use the command pyenv install 3.10.17. This might take a while depending on your internet connection. ☕
Rarely, you might encounter issues downloading the specific version. If this happens, try searching for the Python 3.10.17 installer directly on python.org and installing it manually. Then, use pyenv rehash.

7. Set the Global Python Version (Optional)

You can set Python 3.10.17 as your global Python version, meaning it will be used in every terminal window you open. This is convenient, but you can also set it per project (see Step 4).

Use the command: pyenv global 3.10.17
Verify the installation with: python --version 🐍

8. Setting Python Version per Project (Recommended)

This is generally the preferred method, as it keeps your projects isolated and avoids conflicts. This is done using pyenv local.

Navigate to your project directory using the command line. 📁
Run pyenv local 3.10.17. This creates a .python-version file in your directory specifying the Python version.
Now, whenever you're in this project directory, pyenv will automatically use Python 3.10.17.

9. Install Python dependencies for ML

```
pip install torch torchvision torchaudio
```

10. Install the JFrog ML SDK

```
pip install qwak-sdk
```

## Next Steps

Now that you have set up the environment, you can start using the JFrog ML SDK to build, train, and deploy ML models. For more information, refer to the [JFrog ML Documentation](https://docs.qwak.com/docs/introduction).
