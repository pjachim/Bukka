# Bukka CLI Guide

This guide covers the enhanced CLI features for Bukka, the ML Project Scaffolding Tool.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Commands](#commands)
  - [init-config](#init-config)
  - [run](#run)
- [Configuration File](#configuration-file)
- [Examples](#examples)
- [Input Validation](#input-validation)

## Overview

The Bukka CLI provides a robust interface for creating machine learning project scaffolds with automatic pipeline generation. It supports:

- ✅ YAML configuration files
- ✅ Multiple dataframe backends (Polars, Pandas, Modin, cuDF, Dask, PyArrow)
- ✅ Problem type specification (classification, regression, clustering)
- ✅ Comprehensive input validation
- ✅ Detailed help documentation

## Installation

```bash
pip install -e .
```

Make sure PyYAML is installed (it's included in the dependencies).

## Commands

### init-config

Create a YAML configuration template with default values and documentation.

**Usage:**
```bash
python -m bukka init-config [--output OUTPUT]
```

**Arguments:**
- `--output, -o`: Output path for the config template (default: `bukka_config.yaml`)

**Example:**
```bash
python -m bukka init-config
python -m bukka init-config --output my_project_config.yaml
```

### run

Create and set up a Bukka ML project with automatic pipeline generation.

**Usage:**
```bash
python -m bukka run [OPTIONS]
```

**Configuration Options:**
- `--config, -c`: Path to YAML configuration file (overrides other arguments)

**Project Settings:**
- `--name, -n`: Project name / directory to create (required unless using --config)
- `--dataset, -d`: Path to dataset file (CSV, Parquet, etc.)
- `--target, -t`: Name of the target column (omit for clustering)
- `--skip-venv, -sv`: Skip virtual environment creation

**Data Processing:**
- `--backend, -b`: Dataframe backend (default: polars)
  - Choices: `polars`, `pandas`, `modin`, `cudf`, `dask`, `pyarrow`
- `--train-size`: Train/test split ratio (default: 0.8)
- `--stratify`: Stratify train/test split (default: True)
- `--no-stratify`: Disable stratified splitting
- `--strata`: Column(s) to use for stratification

**Problem Specification:**
- `--problem-type, -p`: ML problem type (default: auto)
  - Choices: `binary_classification`, `multiclass_classification`, `regression`, `clustering`, `auto`

## Configuration File

The YAML configuration file provides a convenient way to manage project settings. Generate a template with:

```bash
python -m bukka init-config
```

**Example `bukka_config.yaml`:**
```yaml
# Bukka Project Configuration Template

# Project settings
project:
  name: my_ml_project
  dataset: data/train.csv
  target: target_column
  skip_venv: false

# Data processing settings
data:
  backend: polars
  train_size: 0.8
  stratify: true
  strata: null

# Problem specification
problem:
  type: auto  # or: binary_classification, multiclass_classification, regression, clustering
```

**Using the config file:**
```bash
python -m bukka run --config bukka_config.yaml
```

**Override config values with CLI arguments:**
```bash
python -m bukka run --config bukka_config.yaml --backend pandas --problem-type regression
```

## Examples

### 1. Quick Start - Create a project with inline arguments

```bash
python -m bukka run --name my_project --dataset data.csv --target price
```

### 2. Specify backend and problem type

```bash
python -m bukka run -n classification_proj -d iris.csv -t species \
  --backend pandas --problem-type multiclass_classification
```

### 3. Regression project with custom train/test split

```bash
python -m bukka run -n regression_proj -d housing.csv -t price \
  --problem-type regression --train-size 0.7
```

### 4. Clustering project (no target column)

```bash
python -m bukka run -n clustering_proj -d customers.csv \
  --problem-type clustering --backend polars
```

### 5. Create project structure only (add dataset later)

```bash
python -m bukka run --name future_project --skip-venv
```

### 6. Use configuration file

```bash
# First, create a config template
python -m bukka init-config --output my_config.yaml

# Edit my_config.yaml with your settings

# Run with the config
python -m bukka run --config my_config.yaml
```

### 7. Stratified sampling with specific columns

```bash
python -m bukka run -n stratified_proj -d data.csv -t outcome \
  --strata gender age_group --backend polars
```

## Input Validation

The CLI includes comprehensive input validation:

### ✅ Project Name Validation
- Cannot be empty
- Cannot contain invalid characters: `< > : " | ? *`

```bash
# ❌ Invalid
python -m bukka run --name "my:project"
# Error: Project name contains invalid characters

# ✅ Valid
python -m bukka run --name my_project
```

### ✅ Dataset Path Validation
- File must exist
- Path must point to a file (not a directory)

```bash
# ❌ Invalid
python -m bukka run --name proj --dataset nonexistent.csv
# Error: Dataset file not found: nonexistent.csv

# ✅ Valid
python -m bukka run --name proj --dataset ./data/train.csv
```

### ✅ Backend Validation
- Must be a narwhals-supported backend

```bash
# ❌ Invalid
python -m bukka run --name proj --backend invalid
# Error: Backend 'invalid' not supported

# ✅ Valid
python -m bukka run --name proj --backend polars
```

### ✅ Problem Type Validation
- Must be a recognized ML problem type

```bash
# ❌ Invalid
python -m bukka run --name proj --problem-type unsupervised
# Error: Problem type 'unsupervised' not recognized

# ✅ Valid
python -m bukka run --name proj --problem-type clustering
```

### ✅ Train Size Validation
- Must be between 0 and 1

```bash
# ❌ Invalid
python -m bukka run --name proj --train-size 1.5
# Error: train_size must be between 0 and 1

# ✅ Valid
python -m bukka run --name proj --train-size 0.8
```

## Help

Get help at any level:

```bash
# Main help
python -m bukka --help

# Subcommand help
python -m bukka run --help
python -m bukka init-config --help
```

## Migration from Old CLI

If you're upgrading from an older version of Bukka, here's how to migrate:

**Old CLI:**
```bash
python -m bukka --name my_proj --dataset data.csv --target label
```

**New CLI (backward compatible, but deprecated):**
```bash
python -m bukka run --name my_proj --dataset data.csv --target label
```

**New CLI (recommended with new features):**
```bash
python -m bukka run --name my_proj --dataset data.csv --target label \
  --backend polars --problem-type auto
```

## Troubleshooting

### ModuleNotFoundError: No module named 'yaml'

Install PyYAML:
```bash
pip install pyyaml
```

### Validation errors

Read the error messages carefully - they tell you exactly what's wrong:
```
✗ Dataset file not found: data.csv
✗ Backend 'numpy' not supported. Supported backends: polars, pandas, modin, cudf, dask, pyarrow
```

### Config file not loading

Make sure your YAML is valid:
```bash
python -c "import yaml; yaml.safe_load(open('bukka_config.yaml'))"
```

## Contributing

Found a bug or want to request a feature? Visit: https://github.com/pjachim/Bukka/issues
