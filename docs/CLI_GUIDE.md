# Bukka CLI Guide

Bukka is a command-line tool for starting a machine learning project without spending time on hand-built scaffolding. The CLI does the routine setup work: it creates the project layout, writes the config files, copies the dataset, and sets up the starting files you are meant to edit.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Commands](#commands)
  - [init-config](#init-config)
  - [run](#run)
- [Configuration file](#configuration-file)
- [Examples](#examples)
- [Help](#help)

## Overview

The main entry point is `python -m bukka run`. The CLI supports:

- YAML configuration files
- Multiple dataframe backends
- Explicit problem type selection
- Optional MLflow setup
- Optional dummy and TPOT helpers
- Train/test split controls

## Installation

```bash
pip install bukka
```

## Commands

### `init-config`

Create a starter YAML configuration file.

```bash
python -m bukka init-config [--output OUTPUT]
```

Arguments:

- `--output`, `-o`: Output path for the generated config file. Defaults to `bukka_config.yaml`.

Examples:

```bash
python -m bukka init-config
python -m bukka init-config --output my_project_config.yaml
```

### `run`

Create and set up a Bukka project.

```bash
python -m bukka run [OPTIONS]
```

Configuration source:

- `--config`, `-c`: Path to a YAML config file.

Project settings:

- `--name`, `-n`: Project name or directory.
- `--dataset`, `-d`: Path to the dataset file.
- `--target`, `-t`: Target column name.
- `--skip-venv`, `-sv`: Skip virtual environment creation.
- `--mlflow`: Enable MLflow setup.
- `--mlflow-tracking-uri`: Custom MLflow tracking URI.

Data settings:

- `--backend`, `-b`: Dataframe backend.
- `--train-size`: Train/test split ratio.
- `--stratify`: Enable stratified splitting.
- `--no-stratify`: Disable stratified splitting.
- `--strata`: Column or columns used for stratification.

Project helpers:

- `--dummy`: Write a dummy baseline helper.
- `--tpot`: Write a TPOT helper.

Problem settings:

- `--problem-type`, `-p`: Problem type.

The CLI currently accepts the backend names supported by the codebase, including `polars`, `pandas`, `modin`, `cudf`, `dask`, and `pyarrow`. The problem type choices are the ones exposed by Bukka itself.

## Configuration file

Use a YAML file when the command line gets too long or when you want to reuse the same setup.

Generate a template:

```bash
python -m bukka init-config
```

Example file:

```yaml
project:
  name: my_ml_project
  dataset: data/train.csv
  target: target_column
  skip_venv: false
  enable_mlflow: false
  mlflow_tracking_uri: null

data:
  backend: polars
  train_size: 0.8
  stratify: true
  strata: null

problem:
  type: auto
```

Run with a config file:

```bash
python -m bukka run --config bukka_config.yaml
```

You can still override values on the command line:

```bash
python -m bukka run --config bukka_config.yaml --backend pandas --train-size 0.7
```

## Examples

Create a project from a CSV file:

```bash
python -m bukka run --name titanic --dataset titanic.csv --target Survived
```

Use a different backend and problem type:

```bash
python -m bukka run -n fraud_detection -d transactions.csv -t is_fraud \
  --backend pandas --problem-type binary_classification
```

Create a project without the virtual environment:

```bash
python -m bukka run -n quick_start -d data.csv --skip-venv
```

Add MLflow or helper files when you need them:

```bash
python -m bukka run -n tracked_project -d data.csv -t target --mlflow
python -m bukka run -n baseline_project -d data.csv -t target --dummy
python -m bukka run -n tpot_project -d data.csv -t target --tpot
```

## Help

Get help from the CLI itself:

```bash
python -m bukka --help
python -m bukka run --help
python -m bukka init-config --help
```

## Notes

- `run` is the main command for day-to-day use.
- `init-config` is there for people who prefer YAML over long command lines.
- The deeper module-level API is documented in the Sphinx docs and is mainly for advanced users.
