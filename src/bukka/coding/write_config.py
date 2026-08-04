from bukka.utils.files.file_manager import FileManager
from pathlib import Path
CONFIG_TEMPLATE = """
from pathlib import Path

DATAFRAME_BACKEND = '{backend_name}'

# Current Directory
CURRENT_DIR = Path(__file__).parent

# Dataset Paths
TRAIN_DATASET_PATH = CURRENT_DIR / '{train_relative_path}'
TEST_DATASET_PATH = CURRENT_DIR / '{test_relative_path}'
"""

class ConfigWriter:
    """
    Generates and writes a configuration file for a Bukka project.

    This class constructs a Python source file containing configuration settings
    for the Bukka project, such as the DataFrame backend to use and optional
    MLflow configuration.

    Parameters
    ----------
    output_path : str
        The file path where the configuration file will be written.
    backend_name : str
        The name of the DataFrame backend to use with Narwhals (e.g., 'pyarrow', 'modin', 'cudf', 'dask').
    file_manager : FileManager
        Manager for project file paths and directory structure.
    enable_mlflow : bool, optional
        Whether to include MLflow configuration (default: False).
    mlflow_tracking_uri : str | None, optional
        MLflow tracking URI. If None and enable_mlflow is True,
        defaults to mlruns directory (default: None).

    Examples
    --------
    >>> writer = ConfigWriter(output_path="config.py", backend_name="pyarrow", file_manager=fm)
    >>> writer.write_config()  # Writes the config file with pyarrow backend
    """
    def __init__(
        self,
        output_path: str,
        backend_name: str,
        file_manager: FileManager,
        enable_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None
    ):
        self.output_path = output_path
        self.backend_name = backend_name
        self.file_manager = file_manager
        self.enable_mlflow = enable_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def write_config(self) -> None:
        """
        Write the configuration file to the configured output path.

        Generates Python source code from the template and writes it to
        the specified file. Conditionally includes MLflow configuration.

        Examples
        --------
        >>> writer = ConfigWriter(output_path="config.py", backend_name="pyarrow", file_manager=fm)
        >>> writer.write_config()  # Writes the config file with pyarrow backend
        """
        config_code = CONFIG_TEMPLATE.format(
            backend_name=self.backend_name,
            train_relative_path=self.file_manager.train_file_relative_path.as_posix(),
            test_relative_path=self.file_manager.test_file_relative_path.as_posix()
        )
        
        # Append MLflow configuration if enabled
        if self.enable_mlflow:
            if self.mlflow_tracking_uri is None:
                tracking_uri = f"sqlite:///{(Path(*self.file_manager.mlruns_path.parts[1:]) / 'mlflow.db').as_posix()}"
            else:
                tracking_uri = self.mlflow_tracking_uri
            
            mlflow_config = f'''
# MLflow Configuration
MLFLOW_TRACKING_URI = "{tracking_uri}"
MLFLOW_EXPERIMENT_NAME = "{self.file_manager.project_path.name}_experiment"
'''
            config_code += mlflow_config
        
        with open(self.output_path, 'w') as file:
            file.write(config_code)