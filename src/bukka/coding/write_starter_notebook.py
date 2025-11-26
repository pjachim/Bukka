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

    Examples
    --------
    >>> writer = StarterNotebookWriter(output_path="starter_notebook.ipynb")
    >>> writer.write_notebook()  # Writes the starter notebook to file
    >>> 
    >>> # With virtual environment
    >>> writer = StarterNotebookWriter(
    ...     output_path="starter_notebook.ipynb",
    ...     venv_path=".venv"
    ... )
    >>> writer.write_notebook()  # Writes notebook configured for the venv
    """
    def __init__(self, output_path: str, venv_path: str | Path | None = None) -> None:
        self.output_path = output_path
        self.venv_path = venv_path

    def write_notebook(self) -> None:
        """
        Write the starter Jupyter notebook to the configured output path.
        """
        with JupyterWriter(self.output_path, venv_path=self.venv_path) as notebook_writer:
            notebook_writer.add_cell(
                cell_content="# Welcome to Your Bukka Project\n\nThis notebook will help you get started with your Bukka project.",
                cell_type="markdown"
            )

            notebook_writer.add_cell(
                cell_content="## Data Loading\n\nThe following code snippet demonstrates how to load your training and testing data using the `DataReader` class provided by Bukka.",
                cell_type="markdown"
            )
            notebook_writer.add_cell(
                cell_content=(
                    "# Import necessary libraries\n"
                    "import pandas as pd\n"
                    "from utils.data_reader import DataReader\n\n"
                    "# Load your data\n"
                    "data_reader = DataReader()\ntrain_data = data_reader.read_train_data()\ntest_data = data_reader.read_test_data()\n\n"
                    "# Display the first few rows of the training data\ntrain_data.head()"
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
                    "from pipelines.generated.pipeline_TIMESTAMP import get_pipeline\n\n"
                    "# Get the pipeline instance\n"
                    "pipeline = get_pipeline()\n\n"
                    "# Fit the pipeline on training data\n"
                    "pipeline.fit(train_data.drop(columns=['target']), train_data['target'])\n\n"
                    "# Make predictions on test data\n"
                    "predictions = pipeline.predict(test_data.drop(columns=['target']))\n\n"
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
                    "accuracy = accuracy_score(test_data['target'], predictions)\n"
                    "print(f'Accuracy: {accuracy:.4f}')\n\n"
                    "# Display detailed classification report\n"
                    "print('\\nClassification Report:')\n"
                    "print(classification_report(test_data['target'], predictions))"
                ),
                cell_type="code"
            )