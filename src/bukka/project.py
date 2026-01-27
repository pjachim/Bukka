from datetime import datetime
from pathlib import Path
from typing import Optional

from bukka.utils.files.file_manager import FileManager
from bukka.environment.environment import EnvironmentBuilder
from bukka.data_management.dataset import Dataset
from bukka.coding.write_pipeline import PipelineWriter
from bukka.coding.write_data_reader_class import DataReaderWriter
from bukka.coding.write_starter_notebook import StarterNotebookWriter
from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
from bukka.coding.write_config import ConfigWriter
from bukka.utils.bukka_logger import BukkaLogger
from bukka.expert_system.pipeline_builder import PipelineBuilder

logger = BukkaLogger(__name__)

class Project:
    """Represents a data science or ML project, managing its file structure and environment setup.
    
    This class orchestrates project creation, environment setup, and pipeline generation
    for machine learning projects.
    
    Parameters
    ----------
    name : str
        The name of the project (used as the project path).
    dataset_path : str | None
        The path to the original dataset file (optional).
    target_column : str | None
        The name of the target column in the dataset (optional).
    skip_venv : bool, optional
        Whether to skip virtual environment creation. Defaults to False.
    enable_mlflow : bool, optional
        Whether to enable MLflow experiment tracking. Defaults to False.
    mlflow_tracking_uri : str | None, optional
        MLflow tracking URI (optional). Defaults to file-based tracking in mlruns/.
    backend : str, optional
        Dataframe backend to use via Narwhals (e.g., 'polars', 'pyarrow', 'modin', 'cudf', 'dask'). Defaults to 'polars'.
    problem_type : str, optional
        ML problem type specification. Defaults to 'auto'.
    train_size : float, optional
        Proportion of data for training split. Defaults to 0.8.
    stratify : bool, optional
        Whether to stratify the train/test split. Defaults to True.
    strata : list[str] | None, optional
        Column(s) to use for stratification. Defaults to None.
        
    Examples
    --------
    >>> proj = Project(
    ...     name="my_project",
    ...     dataset_path="data.csv",
    ...     target_column="target",
    ...     backend="pyarrow",
    ...     problem_type="binary_classification"
    ... )
    >>> proj.run()
    
    >>> # With MLflow enabled
    >>> proj = Project(
    ...     name="tracked_project",
    ...     dataset_path="data.csv",
    ...     target_column="target",
    ...     enable_mlflow=True
    ... )
    >>> proj.run()
    """
    def __init__(
            self,
            name: str,
            dataset_path: str | None = None,
            target_column: str | None = None,
            skip_venv: bool = False,
            enable_mlflow: bool = False,
            mlflow_tracking_uri: str | None = None,
            backend: str = "polars",
            problem_type: str = "auto",
            train_size: float = 0.8,
            stratify: bool = True,
            strata: list[str] | None = None
        ) -> None:
        """Initialize a Project instance.

        Args:
            name: The name of the project (used as the project path).
            dataset_path: The path to the original dataset file (optional).
            target_column: The name of the target column (optional).
            skip_venv: Whether to skip virtual environment creation.
            enable_mlflow: Whether to enable MLflow experiment tracking.
            mlflow_tracking_uri: MLflow tracking URI (optional).
            backend: Dataframe backend to use (default: 'polars').
            problem_type: ML problem type (default: 'auto').
            train_size: Train/test split ratio (default: 0.8).
            stratify: Whether to stratify the split (default: True).
            strata: Column(s) for stratification (default: None).
        """
        logger.info(f"Initializing Project: '{name}'")
        logger.debug(f"Dataset path: {dataset_path}")
        logger.debug(f"Target column: {target_column}")
        logger.debug(f"Backend: {backend}")
        logger.debug(f"Problem type: {problem_type}")
        logger.debug(f"MLflow enabled: {enable_mlflow}")
        
        self.name: str = name
        self.dataset_path: str | None = dataset_path
        self.file_manager: FileManager | None = None
        self.target_column: str | None = target_column
        self.environ_manager: EnvironmentBuilder | None = None
        self.skip_venv: bool = skip_venv
        self.enable_mlflow: bool = enable_mlflow
        self.mlflow_tracking_uri: str | None = mlflow_tracking_uri
        self.backend: str = backend
        self.problem_type: str = problem_type
        self.train_size: float = train_size
        self.stratify: bool = stratify
        self.strata: list[str] | None = strata
        
        logger.debug("Project instance created")

    def run(self) -> None:
        """
        Run the project setup: build the file skeleton and set up the environment.
        """
        logger.info(f"Running project setup for '{self.name}'", format_level='h3')
        
        logger.info("Building project skeleton")
        self._build_skeleton()
        
        if not self.skip_venv:
            logger.info("Setting up project environment")
            self._write_toml()
            self._setup_environment()
        else:
            logger.info("Skipping environment setup as per configuration")

        if self.dataset_path:
            logger.info("Dataset path provided, generating pipeline")
            self._write_pipeline(
                target_column=self.target_column,
                dataframe_backend=self.backend,
                strata=self.strata,
                stratify=self.stratify
            )
            self._write_data_reader_class()
            self._write_config()
            
            if self.enable_mlflow:
                logger.info("MLflow enabled, generating MLflow setup")
                self._write_mlflow_setup()
            
            self._write_starter_notebook()
        else:
            logger.debug("No dataset path provided, skipping pipeline generation")
        
        logger.info(f"Project setup complete for '{self.name}'", format_level='h4')

    def _write_pipeline(
            self,
            target_column: str,
            dataframe_backend: str = "polars",
            strata: list[str] | None = None,
            stratify: bool = True,
        ):
        """Generate a candidate pipeline and save it to the project pipelines folder.

        This method creates a `Dataset` using the project's `FileManager`, runs
        the expert system `ProblemIdentifier` to detect problems and select
        solutions, and then uses `PipelineWriter` to produce pipeline code.

        The resulting pipeline text is written to a timestamped file under
        `FileManager.generated_pipes` and the file path is returned.

        Args:
            target_column: Name of the target column in the dataset (pass
                `None` only if clustering is intended and the Dataset
                backend supports a None target — otherwise provide the
                appropriate column name).
            dataframe_backend: The dataframe backend to use when creating
                the `Dataset` (default: `'polars'`).

        Returns:
            The absolute path (string) of the written pipeline file.
        """
        logger.info("Starting pipeline generation", format_level='h4')
        logger.debug(f"Target column: {target_column}")
        logger.debug(f"Dataframe backend: {dataframe_backend}")

        logger.info("Creating Dataset instance")
        dataset = Dataset(
            target_column, 
            self.file_manager,
            strata=strata,
            stratify=stratify,
            train_size=self.train_size,
            backend=dataframe_backend
        )
        logger.debug("Dataset instance created")
        builder = PipelineBuilder(dataset, target_column, problem_type=self.problem_type)
        pipeline_steps = builder.build_pipeline()

        # Generate pipeline
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"pipeline_{timestamp}.py"

        logger.info("Generating pipeline code")
        writer = PipelineWriter(
            pipeline_steps=pipeline_steps,
            output_path=self.file_manager.generated_pipes / filename
        )
        writer.write_code()
        logger.debug(f"Pipeline written to: {self.file_manager.generated_pipes / filename}")
        logger.info("Pipeline generation complete", format_level='h4')
    
    def _write_data_reader_class(self) -> None:
        """Generate and write a data reader class to the project.

        This method creates a data reader class that encapsulates
        the logic for loading the dataset, using the project's  `FileManager`.
        The generated class is saved to the project's data readers folder.  
        """
        writer = DataReaderWriter(self.file_manager, target_column=self.target_column)
        writer.write_code()
        logger.info("Data reader class generation complete", format_level='h4')

    def _write_config(self) -> None:
        """Generate and write a configuration file for the project.

        This method creates a configuration file (e.g., `config.py`) that
        contains settings for the project, such as the DataFrame backend to use
        and optional MLflow configuration.
        """
        writer = ConfigWriter(
            output_path=self.file_manager.config_path,
            backend_name=self.backend,
            file_manager=self.file_manager,
            enable_mlflow=self.enable_mlflow,
            mlflow_tracking_uri=self.mlflow_tracking_uri
        )
        writer.write_config()
        logger.info("Configuration file generation complete", format_level='h4')
    
    def _write_mlflow_setup(self) -> None:
        """Generate and write MLflow setup file for the project.

        This method creates an MLflow setup file that configures experiment
        tracking for the project. The file includes tracking URI configuration
        and experiment naming.
        """
        from bukka.coding.write_mlflow_setup import MLflowSetupWriter
        
        writer = MLflowSetupWriter(
            file_manager=self.file_manager,
            project_name=self.name,
            tracking_uri=self.mlflow_tracking_uri
        )
        writer.write_code()
        
        # Create mlruns directory
        self.file_manager.mlruns_path.mkdir(exist_ok=True)
        
        logger.info("MLflow setup file generation complete", format_level='h4')

    def _build_skeleton(self) -> None:
        """
        Build the project file skeleton using FileManager.
        """
        logger.debug("Initializing FileManager")
        logger.debug(f"Project path: {self.name}")
        logger.debug(f"Original dataset: {self.dataset_path}")
        
        self.file_manager = FileManager(
            project_path=self.name,
            orig_dataset=self.dataset_path
        )
        logger.debug("FileManager initialized")
        
        logger.info("Building project file skeleton")
        self.file_manager.build_skeleton()
        logger.info("Project skeleton built successfully")

    def _setup_environment(self) -> None:
        """
        Set up the project environment using EnvironmentBuilder.
        """
        logger.debug("Setting up project environment")
        
        if self.file_manager is None:
            logger.error("FileManager is None, cannot set up environment")
            raise RuntimeError("FileManager must be initialized before setting up the environment.")
        
        logger.debug("Initializing EnvironmentBuilder")
        self.environ_manager = EnvironmentBuilder(
            file_manager=self.file_manager,
            enable_mlflow=self.enable_mlflow
        )
        logger.debug("EnvironmentBuilder initialized")
        
        logger.info("Building project environment (virtualenv and dependencies)")
        self.environ_manager.build_environment()
        logger.info("Environment setup complete")

    def _write_starter_notebook(self) -> None:
        """
        Generate and write a starter Jupyter notebook for the project.

        This method creates a Jupyter notebook with pre-defined cells
        to help users get started with their Bukka project. If a virtual
        environment was created, the notebook will be configured to use it.
        """
        # Pass venv path if environment was set up
        venv_path = None if self.skip_venv else self.file_manager.virtual_env
        
        starter_notebook_writer = StarterNotebookWriter(
            output_path=str(self.file_manager.starter_notebook_path),
            venv_path=venv_path,
            target_column=self.target_column,
            problem_type=self.problem_type
        )

        logger.info("Writing starter notebook")
        starter_notebook_writer.write_notebook()

    def _write_toml(self) -> None:
        """
        Write a pyproject.toml file for the project.

        This method creates a pyproject.toml file with basic project
        metadata and configuration.
        """
        toml_path = self.file_manager.pyproject_toml_path

        logger.info(f"Writing pyproject.toml to: {toml_path}")
        writer = PyprojectTomlWriter(
            file_manager=self.file_manager,
            project_name=self.name
        )

        writer.write_code()
        logger.info(f"pyproject.toml written to: {toml_path}")