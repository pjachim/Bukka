from pathlib import Path
from bukka.coding.utils.jupyter_handler import JupyterWriter

class StarterNotebookWriter:
    """
    Generates and writes a starter Jupyter notebook for a Bukka project.

    This class constructs a Python source file containing a Jupyter notebook
    with pre-defined cells to help users get started with their Bukka project.

    Parameters
    ----------
    output_path : str
        The file path where the notebook will be written.
    venv_path : str | Path | None, optional
        The path to the virtual environment. If provided, the notebook will be
        configured to use the Python interpreter from this environment.
    target_column : str | None, optional
        The name of the target column for supervised learning tasks. If None,
        generates notebook code for unsupervised learning.
    problem_type : str, optional
        The type of ML problem ('regression', 'classification', 'auto', etc.).
        Defaults to 'auto'.
    enable_mlflow : bool, optional
        Whether to include MLflow experiment tracking examples. Defaults to False.

    Examples
    --------
    >>> writer = StarterNotebookWriter(output_path="starter_notebook.ipynb")
    >>> writer.write_notebook()  # Writes the starter notebook to file
    >>> 
    >>> # With virtual environment and supervised learning
    >>> writer = StarterNotebookWriter(
    ...     output_path="starter_notebook.ipynb",
    ...     venv_path=".venv",
    ...     target_column="target",
    ...     problem_type="regression",
    ...     enable_mlflow=True
    ... )
    >>> writer.write_notebook()  # Writes notebook configured for the venv
    """
    def __init__(
        self,
        output_path: str,
        venv_path: str | Path | None = None,
        target_column: str | None = None,
        problem_type: str = "auto",
        enable_mlflow: bool = False
    ) -> None:
        self.output_path = output_path
        self.venv_path = venv_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.enable_mlflow = enable_mlflow

    def write_notebook(self) -> None:
        """
        Write the starter Jupyter notebook to the configured output path.
        """
        with JupyterWriter(self.output_path, venv_path=self.venv_path) as notebook_writer:
            notebook_writer.add_cell(
                cell_content="# Welcome to Your Bukka Project\n\nThis notebook will help you get started with your Bukka project.",
                cell_type="markdown"
            )

            if self.target_column is not None:
                self._add_supervised_cells(notebook_writer)
            else:
                self._add_unsupervised_cells(notebook_writer)
            
            if self.enable_mlflow:
                self._add_mlflow_cells(notebook_writer)

    def _add_supervised_cells(self, notebook_writer: JupyterWriter) -> None:
        """Add cells for supervised learning tasks."""
        notebook_writer.add_cell(
            cell_content="## Data Loading\n\nThe following code snippet demonstrates how to load your training and testing data using the `DataReader` class provided by Bukka.",
            cell_type="markdown"
        )
        notebook_writer.add_cell(
            cell_content=(
                "# Import necessary libraries\n"
                "from utils.data_reader import DataReader\n\n"
                "# Load your data (X = features, y = target)\n"
                "data_reader = DataReader()\n"
                "X_train, y_train = data_reader.readXy_train()\n"
                "X_test, y_test = data_reader.readXy_test()\n\n"
                "# Display the first few rows of the training data\n"
                "display(X_train)"
            ),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="## Running a Pipeline\n\nBukka generates ML pipelines in the `pipelines/generated/` directory. You can import and run these pipelines to train and evaluate your models.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=(
                "# Import the generated pipeline\n"
                "# Replace 'pipeline_TIMESTAMP' with your actual pipeline filename\n"
                "from pipelines.generated.pipeline_TIMESTAMP import pipeline\n\n"
                "# Fit the pipeline on training data\n"
                "pipeline.fit(X_train, y_train)\n\n"
                "# Make predictions on test data\n"
                "predictions = pipeline.predict(X_test)\n\n"
                "# Display predictions\n"
                "print(predictions[:10])"
            ),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="## Model Evaluation\n\nEvaluate your model's performance using appropriate metrics.",
            cell_type="markdown"
        )

        # Add regression or classification metrics based on problem type
        if self.problem_type.lower() == "regression":
            self._add_regression_evaluation_cell(notebook_writer)
        else:
            self._add_classification_evaluation_cell(notebook_writer)

    def _add_classification_evaluation_cell(self, notebook_writer: JupyterWriter) -> None:
        """Add classification evaluation metrics cell."""
        notebook_writer.add_cell(
            cell_content=(
                "# Import evaluation metrics\n"
                "from sklearn.metrics import accuracy_score, classification_report\n\n"
                "# Calculate accuracy (adjust metric based on your problem type)\n"
                "accuracy = accuracy_score(y_test, predictions)\n"
                "print(f'Accuracy: {accuracy:.4f}')\n\n"
                "# Display detailed classification report\n"
                "print('\\nClassification Report:')\n"
                "print(classification_report(y_test, predictions))"
            ),
            cell_type="code"
        )

    def _add_regression_evaluation_cell(self, notebook_writer: JupyterWriter) -> None:
        """Add regression evaluation metrics cell."""
        notebook_writer.add_cell(
            cell_content=(
                "# Import evaluation metrics\n"
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
                "import numpy as np\n\n"
                "# Calculate regression metrics\n"
                "mae = mean_absolute_error(y_test, predictions)\n"
                "mse = mean_squared_error(y_test, predictions)\n"
                "rmse = np.sqrt(mse)\n"
                "r2 = r2_score(y_test, predictions)\n\n"
                "# Display regression metrics\n"
                "print(f'Mean Absolute Error (MAE): {mae:.4f}')\n"
                "print(f'Mean Squared Error (MSE): {mse:.4f}')\n"
                "print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')\n"
                "print(f'R² Score: {r2:.4f}')"
            ),
            cell_type="code"
        )

    def _add_unsupervised_cells(self, notebook_writer: JupyterWriter) -> None:
        """Add cells for unsupervised learning tasks."""
        notebook_writer.add_cell(
            cell_content="## Data Loading\n\nThe following code snippet demonstrates how to load your training and testing data using the `DataReader` class provided by Bukka.",
            cell_type="markdown"
        )
        notebook_writer.add_cell(
            cell_content=(
                "# Import necessary libraries\n"
                "from utils.data_reader import DataReader\n\n"
                "# Load your data\n"
                "data_reader = DataReader()\n"
                "train_data = data_reader.read_train_data()\n"
                "test_data = data_reader.read_test_data()\n\n"
                "# Display the first few rows of the training data\n"
                "# Convert to pandas for display (Narwhals DataFrame -> pandas)\n"
                "train_data.to_pandas().head()"
            ),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="## Running a Pipeline\n\nBukka generates ML pipelines in the `pipelines/generated/` directory. You can import and run these pipelines for unsupervised learning tasks.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=(
                "# Import the generated pipeline\n"
                "# Replace 'pipeline_TIMESTAMP' with your actual pipeline filename\n"
                "from pipelines.generated.pipeline_TIMESTAMP import pipeline\n\n"
                "# Fit the pipeline on training data\n"
                "pipeline.fit(train_data)\n\n"
                "# Transform or predict on test data\n"
                "results = pipeline.transform(test_data)\n"
                "# Or for clustering: labels = pipeline.predict(test_data)\n\n"
                "# Display results\n"
                "print(results[:10])"
            ),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="## Exploring Results\n\nFor unsupervised learning, you can explore patterns, clusters, or transformed features.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=(
                "# Example: For clustering, analyze cluster distributions\n"
                "# labels = pipeline.predict(train_data)\n"
                "# import pandas as pd\n"
                "# pd.Series(labels).value_counts().sort_index()\n\n"
                "# Example: For dimensionality reduction, visualize transformed data\n"
                "# import matplotlib.pyplot as plt\n"
                "# transformed = pipeline.transform(train_data)\n"
                "# plt.scatter(transformed[:, 0], transformed[:, 1])\n"
                "# plt.xlabel('Component 1')\n"
                "# plt.ylabel('Component 2')\n"
                "# plt.show()"
            ),
            cell_type="code"
        )

    def _add_mlflow_cells(self, notebook_writer: JupyterWriter) -> None:
        """Add MLflow experiment tracking cells."""
        notebook_writer.add_cell(
            cell_content="## MLflow Experiment Tracking\n\nThis project is configured with MLflow for experiment tracking. "
                        "Track your model's hyperparameters, metrics, and artifacts using the `setup_mlflow()` function.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=(
                "# Initialize MLflow tracking with your project's configuration\n"
                "from scripts.mlflow_setup import setup_mlflow\n\n"
                "# Setup MLflow (reads configuration from config.py)\n"
                "mlflow_client = setup_mlflow()\n"
                "print(\"MLflow is now initialized!\")"
            ),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="### Log Your Experiment\n\nWrap your model training code in `mlflow.start_run()` to automatically track parameters and metrics.",
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=(
                "# Example: Training with MLflow tracking\n"
                "with mlflow.start_run(run_name=\"my_first_experiment\"):\n"
                "    # Log hyperparameters\n"
                "    mlflow.log_param(\"model_type\", \"random_forest\")\n"
                "    mlflow.log_param(\"n_estimators\", 100)\n"
                "    mlflow.log_param(\"max_depth\", 10)\n\n"
                "    # Train your model\n"
                "    # model = pipeline.fit(X_train, y_train)\n"
                "    # predictions = pipeline.predict(X_test)\n\n"
                "    # Log metrics\n"
                "    # accuracy = accuracy_score(y_test, predictions)\n"
                "    # mlflow.log_metric(\"accuracy\", accuracy)\n"
                "    # mlflow.log_metric(\"f1_score\", f1_score(y_test, predictions))\n\n"
                "    # Add tags for organization\n"
                "    mlflow.set_tag(\"dataset\", \"your_dataset\")\n"
                "    mlflow.set_tag(\"version\", \"v1\")\n\n"
                "    print(f\"Run ID: {mlflow.active_run().info.run_id}\")"
            ),
            cell_type="code"
        )

        notebook_writer.add_cell(
            cell_content="### View Your Experiments\n\nTo view all your tracked experiments in the MLflow UI, run the following command in your terminal:\n\n"
                        "```bash\n"
                        "mlflow ui\n"
                        "```\n\n"
                        "Then open [http://localhost:5000](http://localhost:5000) in your browser to see:\n"
                        "- All experiment runs\n"
                        "- Logged parameters and metrics\n"
                        "- Performance comparisons across runs\n"
                        "- Experiment artifacts\n\n"
                        "For more details, see the `mlflow_notebook.ipynb` file for a comprehensive tutorial.",
            cell_type="markdown"
        )