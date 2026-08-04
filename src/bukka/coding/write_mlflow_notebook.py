"""MLflow notebook generator for Bukka projects.

This module creates a Jupyter notebook that demonstrates how to use MLflow
with the generated project's dataset, data reader, and baseline dummy model.
"""
from pathlib import Path

from bukka.coding.utils.jupyter_handler import JupyterWriter
from bukka.coding.write_dummy import DUMMY_MAPPING


class MLflowNotebookWriter:
    """
    Generates and writes an MLflow tutorial notebook for a Bukka project.

    This class creates a Jupyter notebook that explains how to use MLflow for
    experiment tracking with the project's mlflow_setup.py script.

    Parameters
    ----------
    output_path : str
        The file path where the notebook will be written.
    venv_path : str | Path | None, optional
        The path to the virtual environment. If provided, the notebook will be
        configured to use the Python interpreter from this environment.

    Examples
    --------
    >>> writer = MLflowNotebookWriter(output_path="mlflow_notebook.ipynb")
    >>> writer.write_notebook()  # Writes the MLflow notebook to file
    """

    def __init__(
        self,
        output_path: str,
        venv_path: str | Path | None = None,
        target_column: str | None = None,
        problem_type: str = "auto"
    ) -> None:
        """Initialize the MLflow notebook writer.
        
        Parameters
        ----------
        output_path : str
            The file path where the notebook will be written.
        venv_path : str | Path | None, optional
            The path to the virtual environment (default: None).
        """
        self.output_path = output_path
        self.venv_path = venv_path
        self.target_column = target_column
        self.problem_type = problem_type

    def write_notebook(self) -> None:
        """Write the MLflow tutorial notebook to the configured output path."""
        with JupyterWriter(self.output_path, venv_path=self.venv_path) as notebook_writer:
            self._add_introduction(notebook_writer)
            self._add_setup_section(notebook_writer)
            self._add_basic_tracking_section(notebook_writer)
            self._add_advanced_tracking_section(notebook_writer)
            self._add_visualization_section(notebook_writer)
            self._add_ui_section(notebook_writer)
            self._add_tips_section(notebook_writer)

    def _add_introduction(self, notebook_writer) -> None:
        """Add introduction cells."""
        notebook_writer.add_cell(
            cell_content="# MLflow Experiment Tracking Tutorial\n\n"
                        "This notebook demonstrates how to use MLflow for tracking machine learning experiments "
                        "with the dataset generated for your Bukka project.\n\n"
                        "**What you'll learn:**\n"
                        "- Setting up MLflow tracking\n"
                        "- Training a baseline dummy model on your project data\n"
                        "- Logging parameters, metrics, and artifacts\n"
                        "- Accessing the MLflow UI",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=self._build_import_cell(),
            cell_type="code"
        )

    def _add_setup_section(self, notebook_writer) -> None:
        """Add MLflow setup section."""
        notebook_writer.add_cell(
            cell_content="## Set Up MLflow\n\n"
                        "Before tracking experiments, you need to initialize MLflow using the `setup_mlflow()` "
                        "function from the `scripts/mlflow_setup.py` file.\n\n"
                        "The MLflow configuration is stored in your `config.py` file with:\n"
                        "- `MLFLOW_TRACKING_URI`: Where to store experiment data\n"
                        "- `MLFLOW_EXPERIMENT_NAME`: Name of your experiment",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content="# Setup MLflow using the project's configuration\n"
                        "from scripts.mlflow_setup import setup_mlflow\n\n"
                        "# Initialize MLflow with project configuration\n"
                        "mlflow_client = setup_mlflow()\n"
                        "print(\"MLflow is now initialized!\")",
            cell_type="code"
        )
    
        notebook_writer.add_cell(
            cell_content="# View the current experiment configuration\n"
                        "print(f\"Tracking URI: {mlflow.get_tracking_uri()}\")\n"
                        "experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)\n"
                        "print(f\"Experiment Name: {experiment.name}\")\n"
                        "print(f\"Experiment ID: {experiment.experiment_id}\")",
            cell_type="code"
        )

    def _add_basic_tracking_section(self, notebook_writer) -> None:
        """Add basic experiment tracking section."""
        notebook_writer.add_cell(
            cell_content="## Experiment Tracking\n\n"
                        "Track a baseline experiment with the split dataset already generated in your project. "
                        "Use `DataReader` to load the train/test data and compare a dummy baseline under MLflow.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=self._build_data_loading_cell(),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content=self._build_basic_run_cell(),
            cell_type="code"
        )

    def _add_advanced_tracking_section(self, notebook_writer) -> None:
        """Add artifact logging examples for the baseline run."""
        notebook_writer.add_cell(
            cell_content="## Artifact Logging\n\n"
                        "MLflow can also store artifacts such as trained models, prediction samples, and figures. "
                        "This cell extends the baseline run with model and dataset artifacts.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=self._build_advanced_run_cell(),
            cell_type="code"
        )

    def _add_visualization_section(self, notebook_writer) -> None:
        """Add result inspection examples."""
        notebook_writer.add_cell(
            cell_content="## Inspect Results\n\n"
                        "Use MLflow search utilities and the tracked metrics to review baseline runs created from your project dataset.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content="# Compare recent runs in the current experiment\n"
                        "runs = mlflow.search_runs(order_by=[\"attributes.start_time DESC\"])\n"
                        "runs[[\"run_id\", \"metrics.test_score\", \"params.model_class\"]].head()",
            cell_type="code"
        )

    def _add_tips_section(self, notebook_writer) -> None:
        """Add closing guidance."""
        notebook_writer.add_cell(
            cell_content="## Next Steps\n\n"
                        "- Swap the dummy baseline for a generated candidate pipeline once you have one.\n"
                        "- Keep the same `DataReader` flow so experiments stay aligned with the project train/test split.\n"
                        "- Add tags such as dataset version or feature set name to make comparisons easier.",
            cell_type="markdown"
        )

    def _build_import_cell(self) -> str:
        """Build the import cell based on the project type."""
        base_imports = [
            "# Import required libraries",
            "import mlflow",
            "import mlflow.sklearn",
            "import pandas as pd",
            "from config import MLFLOW_EXPERIMENT_NAME",
            "from utils.data_reader import DataReader",
        ]

        if self._is_regression():
            base_imports.extend([
                "from sklearn.dummy import DummyRegressor",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
            ])
        else:
            base_imports.extend([
                "from sklearn.dummy import DummyClassifier",
                "from sklearn.metrics import accuracy_score, classification_report",
            ])

        return "\n".join(base_imports)

    def _build_data_loading_cell(self) -> str:
        """Build the data-loading cell for supervised Bukka datasets."""
        return (
            "# Load the train/test split generated by Bukka\n"
            "data_reader = DataReader()\n"
            "X_train, y_train = data_reader.readXy_train(return_pandas=True)\n"
            "X_test, y_test = data_reader.readXy_test(return_pandas=True)\n\n"
            "y_train = y_train.squeeze()\n"
            "y_test = y_test.squeeze()\n\n"
            "print(f\"Training rows: {len(X_train)}\")\n"
            "print(f\"Test rows: {len(X_test)}\")\n"
            "display(X_train.head())"
        )

    def _build_basic_run_cell(self) -> str:
        """Build the baseline MLflow run cell."""
        model_class = self._resolve_dummy_model()
        if self._is_regression():
            metric_block = (
                "    train_predictions = model.predict(X_train)\n"
                "    test_predictions = model.predict(X_test)\n\n"
                "    train_r2 = r2_score(y_train, train_predictions)\n"
                "    test_r2 = r2_score(y_test, test_predictions)\n"
                "    test_mae = mean_absolute_error(y_test, test_predictions)\n"
                "    test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5\n\n"
                "    mlflow.log_metric(\"train_r2\", train_r2)\n"
                "    mlflow.log_metric(\"test_r2\", test_r2)\n"
                "    mlflow.log_metric(\"test_mae\", test_mae)\n"
                "    mlflow.log_metric(\"test_rmse\", test_rmse)\n\n"
                "    mlflow.log_metric(\"train_score\", train_r2)\n"
                "    mlflow.log_metric(\"test_score\", test_r2)\n\n"
                "    print(f\"Train R2: {train_r2:.4f}\")\n"
                "    print(f\"Test R2: {test_r2:.4f}\")\n"
                "    print(f\"Test MAE: {test_mae:.4f}\")\n"
                "    print(f\"Test RMSE: {test_rmse:.4f}\")"
            )
        else:
            metric_block = (
                "    train_predictions = model.predict(X_train)\n"
                "    test_predictions = model.predict(X_test)\n\n"
                "    train_accuracy = accuracy_score(y_train, train_predictions)\n"
                "    test_accuracy = accuracy_score(y_test, test_predictions)\n\n"
                "    mlflow.log_metric(\"train_accuracy\", train_accuracy)\n"
                "    mlflow.log_metric(\"test_accuracy\", test_accuracy)\n\n"
                "    mlflow.log_metric(\"train_score\", train_accuracy)\n"
                "    mlflow.log_metric(\"test_score\", test_accuracy)\n\n"
                "    print(f\"Train Accuracy: {train_accuracy:.4f}\")\n"
                "    print(f\"Test Accuracy: {test_accuracy:.4f}\")\n"
                "    print(classification_report(y_test, test_predictions))"
            )

        return (
            "# Track a baseline dummy model on the project dataset\n"
            f"model = {model_class}()\n\n"
            "with mlflow.start_run(run_name=\"dummy_baseline\"):\n"
            f"    mlflow.log_param(\"model_class\", \"{model_class}\")\n"
            f"    mlflow.log_param(\"problem_type\", \"{self.problem_type}\")\n"
            f"    mlflow.log_param(\"target_column\", \"{self.target_column}\")\n\n"
            "    model.fit(X_train, y_train)\n\n"
            f"{metric_block}\n\n"
            "    print(f\"Run ID: {mlflow.active_run().info.run_id}\")"
        )

    def _build_advanced_run_cell(self) -> str:
        """Build the artifact logging cell."""
        if self._is_regression():
            extra_metrics = (
                "    test_predictions = model.predict(X_test)\n"
                "    residuals = pd.DataFrame({\"actual\": y_test, \"prediction\": test_predictions})\n"
                "    residuals[\"error\"] = residuals[\"actual\"] - residuals[\"prediction\"]\n"
                "    residuals.head().to_csv(\"prediction_sample.csv\", index=False)\n"
            )
        else:
            extra_metrics = (
                "    test_predictions = model.predict(X_test)\n"
                "    prediction_sample = pd.DataFrame({\"actual\": y_test, \"prediction\": test_predictions})\n"
                "    prediction_sample.head().to_csv(\"prediction_sample.csv\", index=False)\n"
            )

        return (
            "# Log artifacts alongside the baseline model\n"
            f"model = {self._resolve_dummy_model()}()\n\n"
            "with mlflow.start_run(run_name=\"dummy_baseline_with_artifacts\"):\n"
            "    model.fit(X_train, y_train)\n"
            "    mlflow.sklearn.log_model(model, artifact_path=\"model\")\n"
            f"{extra_metrics}"
            "    mlflow.log_artifact(\"prediction_sample.csv\")\n"
            "    mlflow.set_tag(\"dataset_split\", \"bukka_train_test\")\n"
            "    mlflow.set_tag(\"notebook\", \"mlflow_notebook\")"
        )

    def _resolve_dummy_model(self) -> str:
        """Resolve the dummy estimator class for the current project."""
        if self.problem_type in DUMMY_MAPPING:
            return DUMMY_MAPPING[self.problem_type]

        return "DummyClassifier"

    def _is_regression(self) -> bool:
        """Return whether the current problem type is regression."""
        return self.problem_type == "regression"

    def _add_ui_section(self, notebook_writer) -> None:
        """Add UI section."""
        notebook_writer.add_cell(
            cell_content="# Running MLFlow UI\n\n"
                        "You can either run this cell, which will start the MLflow server, or you can run the command in your terminal. If you run the cell here, it will keep running until you manually stop it. \n\n"
                        "The MLFlow UI will be available locally [here](http://localhost:5000).",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content="!mlflow ui --port 5000",
            cell_type="code"
        )

