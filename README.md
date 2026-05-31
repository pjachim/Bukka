# bukka

Bukka is a command-line tool for starting a machine learning project without spending the first hour on folder setup. Run it with a project name and, if you have one, a dataset. It creates the project skeleton, writes the config files, splits the data, and leaves you with a usable starting point.

## What it does

- Creates a project folder with the standard Bukka layout.
- Sets up a virtual environment unless you skip it.
- Copies your dataset into the new project and writes train/test splits.
- Writes starter files such as configuration, a data reader, and a notebook.
- Can add optional extras like MLflow, a dummy pipeline, or TPOT.

## Quick start

Install Bukka:

```bash
pip install bukka
```

Create a project:

```bash
python -m bukka run --name titanic --dataset titanic.csv --target Survived
```

If you prefer a config file, generate one first:

```bash
python -m bukka init-config
python -m bukka run --config bukka_config.yaml
```

## Common options

The `run` command accepts the project name, dataset, target column, backend, train size, stratification options, MLflow flags, and model helpers such as `--dummy` and `--tpot`.

Useful examples:

```bash
python -m bukka run -n my_project -d data.csv -t label --backend pandas --problem-type regression
python -m bukka run -n my_project -d data.csv -t target --train-size 0.7
python -m bukka run -n my_project -d data.csv --skip-venv
```

## Resulting project layout

```text
<project_name>/
├── .venv/
├── data/
│   ├── <dataset_name>
│   ├── test/
│   └── train/
├── notebooks/
│   └── starter_notebook.ipynb
├── pipelines/
│   ├── baseline/
│   ├── candidate/
│   └── generated/
├── utils/
├── pyproject.toml
└── requirements.txt
```

## Documentation

The docs are split by task:

- [Getting started](docs/source/getting_started.rst)
- [CLI examples](docs/source/usage_examples.rst)
- [Configuration](docs/source/configuration.rst)
- [Advanced API reference](docs/source/api_reference.rst)

## License

Apache-2.0. See [LICENSE](LICENSE).