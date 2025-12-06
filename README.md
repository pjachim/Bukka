# 📖 bukka: Django-Inspired ML Infrastructure CLI

**bukka** is a Python command-line interface (CLI) tool designed to dramatically reduce the boilerplate and setup time for new Machine Learning (ML) projects. Inspired by the structure and speed of the Django framework's `startproject` command, `bukka` lets you instantly scaffold a robust, standardized, and ready-to-use project infrastructure.

-----

## ✨ Features

  * **Django-Inspired Structure:** Creates a logical, maintainable folder hierarchy optimized for ML workflows (data, models, notebooks, scripts).
  * **Automated Environment Setup:** Automatically generates a Python virtual environment (`.venv`) to isolate your project dependencies.
  * **Dependency Management:** Creates a starting **`requirements.txt`** file with essential ML packages (e.g., NumPy, Pandas, Scikit-learn).
  * **YAML Configuration:** Use configuration files for complex project setups with `init-config` command.
  * **Multiple Backends (Coming Soon!):** Support for various dataframe backends via narwhals (polars~~, pandas, modin, cudf, dask, pyarrow~~).
  * **Problem Type Detection:** Automatic ML problem identification or explicit specification (classification, regression, clustering).
  * **Intelligent Pipeline Generation:** Automatically generates ML pipelines based on dataset analysis.
  * **CLI Simplicity:** Use simple, intuitive commands with subcommands to create a complete project skeleton in seconds.

-----

## 🚀 Quick Start

### 1\. Installation

`bukka` is available on PyPI.

```bash
pip install bukka
```

### 2\. Creating a New Project

Use the `bukka run` command with your desired project name and dataset.

```bash
# Example: Create a new project named 'titanic'
python -m bukka run --name titanic --dataset titanic.csv --target Survived
```

Or use the shorthand options:

```bash
python -m bukka run -n titanic -d titanic.csv -t Survived
```

### 3\. Using Configuration Files

For complex projects, create a YAML configuration template:

```bash
# Generate a config template
python -m bukka init-config

# Edit bukka_config.yaml with your settings, then run:
python -m bukka run --config bukka_config.yaml
```

### 4\. Advanced Options

```bash
# Specify backend and problem type
python -m bukka run -n my_project -d data.csv -t label \
  --backend pandas --problem-type regression

# Custom train/test split
python -m bukka run -n my_project -d data.csv -t target --train-size 0.7

# Skip virtual environment creation (makes command substantially faster)
python -m bukka run -n my_project -d data.csv --skip-venv
```

This command will:

1.  Create the project folder: `titanic/`
2.  Create and configure a virtual environment: `titanic/.venv/`
3.  Generate the initial dependency file: `titanic/requirements.txt`
4.  Install the packages in the requirements.txt
5.  Copy the data file to your data folder
6.  Split the dataset into training and test sets
7.  Analyze your dataset and generate pipelines based on detected problems
8.  Provide placeholder utility classes you can customize
9.  Provide starter notebooks, so you can get to machine learning ASAP

(Coming soon):

10. Initialize MLFlow to track your parameters and results
11. Provide baseline models for comparison (e.g., random guessing)

## 🌳 Standard Project Structure

When you run `python -m bukka run --name <name>`, the following standardized structure is created, ensuring consistency across all your ML projects:

```
<project_name>/
├── .venv/                         # Isolated Python Virtual Environment
├── data/                          # Storage for raw, processed, and external data
│   ├── <dataset_name>             # Original dataset copy
│   ├── test/                      # Test split data
│   └── train/                     # Training split data
├── pipelines/                     # ML Pipelines
│   ├── __init__.py                # Makes 'pipelines' a Python package
│   ├── baseline/                  # Baseline pipelines (e.g. naive classifiers, currently empty)
│   ├── candidate/                 # Your custom pipelines
│   └── generated/                 # Auto-generated pipelines from dataset analysis
├── notebooks/                     # Jupyter notebooks for experimentation
│   └── starter_notebook.ipynb     # Pre-configured notebook to get started
├── utils/                         # Custom utility classes and functions
├── pyproject.toml                 # Project metadata and dependencies
└── requirements.txt               # Dependency list for pip
```

## 📋 CLI Commands

### Available Commands

```bash
python -m bukka --help                    # Show all available commands
python -m bukka init-config --help        # Help for config template generation
python -m bukka run --help                # Help for project creation
```

### Create Configuration Template

```bash
python -m bukka init-config                         # Creates bukka_config.yaml
python -m bukka init-config --output my_config.yaml # Custom output path
```

### Create Project

**Basic Usage:**
```bash
python -m bukka run --name PROJECT_NAME --dataset DATA.csv --target TARGET_COLUMN
```

**Configuration File:**
```bash
python -m bukka run --config bukka_config.yaml
```

**Advanced Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--name` | `-n` | Project name/directory | Required* |
| `--dataset` | `-d` | Path to dataset file | Optional |
| `--target` | `-t` | Target column name | Optional |
| `--config` | `-c` | YAML config file path | None |
| `--backend` | `-b` | Dataframe backend | `polars` |
| `--problem-type` | `-p` | ML problem type | `auto` |
| `--train-size` | | Train/test split ratio | `0.8` |
| `--skip-venv` | `-sv` | Skip venv creation | `False` |
| `--stratify` | | Enable stratified split | `True` |
| `--no-stratify` | | Disable stratified split | - |
| `--strata` | | Stratification columns | None |

\* Required unless using `--config`

**Supported Backends:**
- `polars` (default)
- More coming soon!

**Problem Types:**
- `auto` (default - automatic detection)
- `binary_classification`
- `multiclass_classification`
- `regression`
- `clustering`

-----

## 🎶 About the Name

The library is named after Bukka White, a country blues guitarist whorecorded from the late 30s and 40s, before being rediscovered in the 60s. One of my favorites by him is *Fixin' to Die Blues*. Country blues guitarists use their technical skill with the guitar to back their singing to create a very complete sound, using their thumb to play rhythym and their fingers to play a melodyand harmonize over that giving it the sound of multiple guitarists.

The choice to name the library after a guitarist is inspired, like much of the rest of this project, by the library Django, named for Django Reinhardt, another phenomenal guitarist.

-----

## 🤝 Contributing

We welcome contributions\! If you have suggestions for new structural templates, essential starter packages, or commands, please open an issue or submit a pull request.

-----

## 📄 License

This project is licensed under the Apache License. See the `LICENSE` file for details.