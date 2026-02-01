"""MLflow setup file generator for Bukka projects.

This module creates a setup file for MLflow experiment tracking integration.
"""
from pathlib import Path
from bukka.coding.utils.template_handler import TemplateBaseClass
from bukka.utils.files.file_manager import FileManager


MLFLOW_SETUP_TEMPLATE = '''"""MLflow experiment tracking setup script.

This module provides a utility to initialize MLflow tracking and write configuration
to the project's config.py file.
"""
import mlflow
from pathlib import Path


def setup_mlflow() -> mlflow:
    """Initialize MLflow tracking with configuration from config.py.
    
    Reads MLflow configuration from config.py and sets up MLflow tracking.
    If MLflow configuration is not present in config.py, this function
    will raise an ImportError.
    
    Returns
    -------
    mlflow
        The mlflow module with configured tracking.
    
    Raises
    ------
    ImportError
        If MLFLOW_TRACKING_URI or MLFLOW_EXPERIMENT_NAME are not defined in config.py.
    
    Examples
    --------
    >>> from scripts.mlflow_setup import setup_mlflow
    >>> mlflow_client = setup_mlflow()
    >>> # Start tracking your experiments
    >>> with mlflow.start_run():
    ...     mlflow.log_param("alpha", 0.5)
    ...     mlflow.log_metric("rmse", 0.85)
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import config
    config_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(config_dir))
    
    try:
        from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
    except ImportError as e:
        raise ImportError(
            "MLflow configuration not found in config.py. "
            "Please ensure config.py includes MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME."
        ) from e
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return mlflow


if __name__ == "__main__":
    setup_mlflow()
    # Read and display configuration
    from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
    print(f"MLflow tracking URI: {{MLFLOW_TRACKING_URI}}")
    print(f"Experiment: {{MLFLOW_EXPERIMENT_NAME}}")
    print("\\nTo view experiments, run: mlflow ui")
'''


class MLflowSetupWriter(TemplateBaseClass):
    """Generates MLflow setup file for a Bukka project.
    
    This class creates a Python file that configures MLflow experiment tracking
    with project-specific settings.
    
    Parameters
    ----------
    file_manager : FileManager
        Manager for project file paths and directory structure.
    project_name : str
        Name of the project for experiment naming.
    tracking_uri : str | None, optional
        MLflow tracking URI. If None, defaults to file-based tracking
        in the project's mlruns directory.
    
    Examples
    --------
    >>> from bukka.utils.files.file_manager import FileManager
    >>> file_manager = FileManager(project_path="my_project", orig_dataset=None)
    >>> writer = MLflowSetupWriter(file_manager, "my_project")
    >>> writer.write_code()
    """
    
    def __init__(
        self,
        file_manager: FileManager,
        project_name: str,
        tracking_uri: str | None = None
    ):
        """Initialize the MLflow setup writer.
        
        Parameters
        ----------
        file_manager : FileManager
            Manager for project file paths and directory structure.
        project_name : str
            Name of the project for experiment naming.
        tracking_uri : str | None, optional
            MLflow tracking URI. If None, defaults to file-based tracking
            in the project's mlruns directory (default: None).
        """
        # Default to file-based tracking in mlruns directory
        if tracking_uri is None:
            tracking_uri = f"file:///{file_manager.mlruns_path}"
        
        kwargs = {}
        
        super().__init__(
            template=MLFLOW_SETUP_TEMPLATE,
            output_path=file_manager.mlflow_setup_path,
            kwargs=kwargs,
            expected_args=[]
        )
