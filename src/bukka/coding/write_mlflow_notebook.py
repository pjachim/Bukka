"""MLflow notebook generator for Bukka projects.

This module creates a Jupyter notebook that demonstrates how to use MLflow
with the project's mlflow_setup.py script for experiment tracking.
"""
from pathlib import Path
from bukka.coding.utils.jupyter_handler import JupyterWriter


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
        venv_path: str | Path | None = None
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

    def write_notebook(self) -> None:
        """Write the MLflow tutorial notebook to the configured output path."""
        with JupyterWriter(self.output_path, venv_path=self.venv_path) as notebook_writer:
            self._add_introduction(notebook_writer)
            self._add_setup_section(notebook_writer)
            self._add_basic_tracking_section(notebook_writer)
            self._add_advanced_tracking_section(notebook_writer)
            self._add_visualization_section(notebook_writer)
            self._add_tips_section(notebook_writer)

    def _add_introduction(self, notebook_writer) -> None:
        """Add introduction cells."""
        notebook_writer.add_cell(
            cell_content="# MLflow Experiment Tracking Tutorial\n\n"
                        "This notebook demonstrates how to use MLflow for tracking machine learning experiments "
                        "in your Bukka project.\n\n"
                        "**What you'll learn:**\n"
                        "- Setting up MLflow tracking\n"
                        "- Logging parameters, metrics, and artifacts\n"
                        "- Accessing the MLflow UI",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content="# Import required libraries\n"
                        "import mlflow\n"
                        "from sklearn.datasets import load_iris\n"
                        "from sklearn.model_selection import train_test_split\n"
                        "from sklearn.ensemble import RandomForestClassifier\n"
                        "from sklearn.metrics import accuracy_score, confusion_matrix\n"
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
                        "print(f\"Tracking URI: {mlflow_client.get_tracking_uri()}\")\n"
                        "print(f\"Experiment Name: {mlflow_client.get_experiment(mlflow_client.get_experiment_by_name(mlflow_client.get_experiment(mlflow_client.get_experiment_by_name(mlflow.get_tracking_uri()).experiment_id).name)).name}\")",
            cell_type="code"
        )

    def _add_basic_tracking_section(self, notebook_writer) -> None:
        """Add basic experiment tracking section."""
        notebook_writer.add_cell(
            cell_content="## Experiment Tracking\n\n"
                        "Track your first experiment! Use `mlflow_client.start_run()` to create a new run, "
                        "then log parameters, metrics, and models.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content="# Load a sample dataset\n"
                        "iris = load_iris()\n"
                        "X, y = iris.data, iris.target\n"
                        "X_train, X_test, y_train, y_test = train_test_split(\n"
                        "    X, y, test_size=0.2, random_state=42\n"
                        ")\n\n"
                        "print(f\"Training set size: {X_train.shape[0]}\")\n"
                        "print(f\"Test set size: {X_test.shape[0]}\")",
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="# Example 1: Basic run with parameters and metrics\n"
                        "with mlflow_client.start_run(run_name=\"iris_rf_v1\"):\n"
                        "    # Log hyperparameters\n"
                        "    n_estimators = 100\n"
                        "    max_depth = 5\n"
                        "    mlflow_client.log_param(\"n_estimators\", n_estimators)\n"
                        "    mlflow_client.log_param(\"max_depth\", max_depth)\n\n"
                        "    # Train model\n"
                        "    model = RandomForestClassifier(\n"
                        "        n_estimators=n_estimators,\n"
                        "        max_depth=max_depth,\n"
                        "        random_state=42\n"
                        "    )\n"
                        "    model.fit(X_train, y_train)\n\n"
                        "    # Log metrics\n"
                        "    train_acc = accuracy_score(y_train, model.predict(X_train))\n"
                        "    test_acc = accuracy_score(y_test, model.predict(X_test))\n"
                        "    mlflow_client.log_metric(\"train_accuracy\", train_acc)\n"
                        "    mlflow_client.log_metric(\"test_accuracy\", test_acc)\n\n"
                        "    print(f\"Train Accuracy: {train_acc:.4f}\")\n"
                        "    print(f\"Test Accuracy: {test_acc:.4f}\")\n"
                        "    print(f\"Run ID: {mlflow.active_run().info.run_id}\")",
            cell_type="code"
        )

    def _add_ui_section(self, notebook_writer) -> None:
        """Add visualization and UI section."""
        notebook_writer.add_cell(
            cell_content="# Running MLFlow UI\n\n"
                        "You can either run this cell, which will start the MLflow server, or you can run the command in your terminal. If you run the cell here, it will keep running until you manually stop it. \n\n"
                        "The MLFlow UI will be available locally [here](http://localhost:5000).",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content="!mlflow server --port 5000"
            cell_type="code"
        )

