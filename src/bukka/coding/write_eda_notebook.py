from pathlib import Path
from bukka.coding.utils.jupyter_handler import JupyterWriter

class EDANotebookWriter:
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
        use_pygwalker: bool = False,
        use_df_profiling: bool = False,
    ) -> None:
        self.output_path = output_path
        self.venv_path = venv_path
        self.use_pygwalker = use_pygwalker
        self.use_df_profiling = use_df_profiling

    def write_notebook(self) -> None:
        """
        Write the starter Jupyter notebook to the configured output path.
        """
        with JupyterWriter(self.output_path, venv_path=self.venv_path) as notebook_writer:
            notebook_writer.add_cell(
                cell_content="# Welcome to Your Bukka Project\n\nThis notebook will help you get started with your Bukka project.",
                cell_type="markdown"
            )

            self._load_data_cells(notebook_writer)

            if self.use_df_profiling:
                self._add_df_profiling_cells(notebook_writer)

            if self.use_pygwalker:
                self._add_pygwalker_cells(notebook_writer)

    def _add_df_profiling_cells(self, notebook_writer: JupyterWriter) -> None:
        '''Add DataFrame profiling cells to the notebook.'''
        notebook_writer.add_cell(
            cell_content=(
                "## DataFrame Profiling\n\n"
                "The following code snippet demonstrates how to use pandas-profiling for exploratory data analysis."
            ),
            cell_type="markdown"
        )
        notebook_writer.add_cell(
            cell_content=(
                "# Import necessary libraries\n"
                "from pandas_profiling import ProfileReport\n\n"
                "# Generate a profiling report\n"
                "profile = ProfileReport(df, title='Pandas Profiling Report')\n\n"
                "# Display the report\n"
                "profile.to_notebook_iframe()"
            ),
            cell_type="code"
        )

    def _add_pygwalker_cells(self, notebook_writer: JupyterWriter) -> None:
        '''Add PyGWalker cells to the notebook.'''
        notebook_writer.add_cell(
            cell_content=(
                "## PyGWalker Interactive Data Exploration\n\n"
                "The following code snippet demonstrates how to use PyGWalker for interactive data exploration."
            ),
            cell_type="markdown"
        )

        notebook_writer.add_cell(
            cell_content=(
                '# Import necessary libraries\n'
                'import pygwalker as pyg\n\n'
                '# Launch PyGWalker for interactive data exploration\n'
                'pyg.walk(df)'
            ),
            cell_type="code"
        )

    def _load_data_cells(self, notebook_writer: JupyterWriter) -> None:
        """Add cells for loading data."""
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
                "df = data_reader.read_train_data(return_pandas=True)\n\n"
                "# Display the first few rows of the training data\n"
                "display(df)"
            ),
            cell_type="code"
        )