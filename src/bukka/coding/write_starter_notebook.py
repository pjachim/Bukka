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

    Examples
    --------
    >>> writer = StarterNotebookWriter(output_path="starter_notebook.ipynb")
    >>> writer.write_notebook()  # Writes the starter notebook to file
    >>> 
    >>> # With virtual environment and supervised learning
    >>> writer = StarterNotebookWriter(
    ...     output_path="starter_notebook.ipynb",
    ...     venv_path=".venv",
    ...     target_column="target"
    ... )
    >>> writer.write_notebook()  # Writes notebook configured for the venv
    """
    def __init__(self, output_path: str, venv_path: str | Path | None = None, target_column: str | None = None) -> None:
        self.output_path = output_path
        self.venv_path = venv_path
        self.target_column = target_column

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

    def _add_supervised_cells(self, notebook_writer) -> None:
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
                "# Convert to pandas for display (Narwhals DataFrame -> pandas)\n"
                "X_train.to_pandas().head()"
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

    def _add_unsupervised_cells(self, notebook_writer) -> None:
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