"""MLflow setup file generator for Bukka projects.

This module creates a setup file for MLflow experiment tracking integration.
"""
from pathlib import Path
from bukka.coding.utils.template_handler import TemplateBaseClass
from bukka.utils.files.file_manager import FileManager


MLFLOW_SETUP_TEMPLATE = '''"""MLflow experiment tracking configuration.

This module provides MLflow configuration and setup for tracking machine learning experiments.
"""
import mlflow
from pathlib import Path


# MLflow configuration
MLFLOW_TRACKING_URI = "{tracking_uri}"
MLFLOW_EXPERIMENT_NAME = "{experiment_name}"


def setup_mlflow() -> mlflow:
    """Initialize MLflow tracking with project configuration.
    
    Sets the tracking URI and experiment name for the current project.
    
    Returns
    -------
    mlflow
        The mlflow module with configured tracking.
    
    Examples
    --------
    >>> mlflow_client = setup_mlflow()
    >>> # Start tracking your experiments
    >>> with mlflow.start_run():
    ...     mlflow.log_param("alpha", 0.5)
    ...     mlflow.log_metric("rmse", 0.85)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return mlflow


if __name__ == "__main__":
    setup_mlflow()
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
        
        kwargs = {
            "tracking_uri": tracking_uri,
            "experiment_name": f"{project_name}_experiment"
        }
        
        super().__init__(
            template=MLFLOW_SETUP_TEMPLATE,
            output_path=file_manager.mlflow_setup_path,
            kwargs=kwargs,
            expected_args=["tracking_uri", "experiment_name"]
        )
