from datetime import datetime
from pathlib import Path
from typing import Optional

from bukka.logistics.files.file_manager import FileManager
from bukka.logistics.environment.environment import EnvironmentBuilder
from bukka.data_management.dataset import Dataset
from bukka.expert_system.problem_identifier import ProblemIdentifier
from bukka.coding.write_pipeline import PipelineWriter
from bukka.coding.write_data_reader_class import DataReaderWriter
from bukka.coding.write_starter_notebook import StarterNotebookWriter
from bukka.utils.bukka_logger import BukkaLogger

logger = BukkaLogger(__name__)

class Project:
    """
    Represents a data science or ML project, managing its file structure and environment setup.
    """
    def __init__(
            self,
            name: str,
            dataset_path: str,
            target_column: str,
            skip_venv: bool = False
        ) -> None:
        """
        Initialize a Project instance.

        Args:
            name (str): The name of the project (used as the project path).
            dataset_path (str): The path to the original dataset file.
        """
        logger.info(f"Initializing Project: '{name}'")
        logger.debug(f"Dataset path: {dataset_path}")
        logger.debug(f"Target column: {target_column}")
        self.name: str = name
        self.dataset_path: str = dataset_path
        self.file_manager: FileManager | None = None
        self.target_column: str = target_column
        self.environ_manager: EnvironmentBuilder | None = None
        self.skip_venv: bool = skip_venv
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
            self._setup_environment()
        else:
            logger.info("Skipping environment setup as per configuration")

        if self.dataset_path:
            logger.info("Dataset path provided, generating pipeline")
            self._write_pipeline(target_column=self.target_column)
            self._write_data_reader_class()
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
        ) -> str:
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
        
        if self.file_manager is None:
            logger.debug("FileManager not initialized, building skeleton")
            # Ensure skeleton exists and dataset is copied
            self._build_skeleton()

        logger.info("Creating Dataset instance")
        dataset = Dataset(
            target_column, 
            self.file_manager, 
            dataframe_backend, 
            strata=strata,
            stratify=stratify
        )
        logger.debug(f"Dataset created with {len(dataset.feature_columns)} features")
        
        logger.info("Initializing ProblemIdentifier")
        identifier = ProblemIdentifier(dataset, target_column)
        
        # Run detection phases
        logger.info("Running multivariate problem detection")
        identifier.multivariate_problems()
        logger.debug("Multivariate problem detection complete")
        
        logger.info("Running univariate problem detection")
        identifier.univariate_problems()
        logger.debug("Univariate problem detection complete")
        
        # identify ml problem (may be clustering/regression/classification)
        logger.info("Identifying ML problem type")
        try:
            identifier._identify_ml_problem()
            logger.debug("ML problem identification complete")
        except Exception as e:
            logger.warn(f"Failed to identify ML problem: {e}")
            # If private method naming changes, ignore to avoid crashing here
            pass

        # Generate pipeline
        logger.info("Generating pipeline code")
        writer = PipelineWriter(identifier)
        logger.debug("PipelineWriter initialized")
        
        _, _ = writer.write()
        pipeline_text = writer.pipeline_definition or ""
        logger.debug(f"Pipeline code generated: {len(pipeline_text)} characters")

        # Prepare destination file
        gen_dir: Path = self.file_manager.generated_pipes
        logger.debug(f"Pipeline destination directory: {gen_dir}")
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"pipeline_{timestamp}.py"
        dest = gen_dir / filename
        logger.debug(f"Pipeline filename: {filename}")

        # Write pipeline text
        logger.info(f"Writing pipeline to: {dest}")
        dest.write_text(pipeline_text, encoding="utf-8")
        logger.info("Pipeline generation complete", format_level='h4')

        return str(dest.resolve())
    
    def _write_data_reader_class(self) -> None:
        """Generate and write a data reader class to the project.

        This method creates a data reader class that encapsulates
        the logic for loading the dataset, using the project's  `FileManager`.
        The generated class is saved to the project's data readers folder.  
        """
        writer = DataReaderWriter(self.file_manager)
        writer.write_class()
        logger.info("Data reader class generation complete", format_level='h4')

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
            file_manager=self.file_manager
        )
        logger.debug("EnvironmentBuilder initialized")
        
        logger.info("Building project environment (virtualenv and dependencies)")
        self.environ_manager.build_environment()
        logger.info("Environment setup complete")

    def _write_starter_notebook(self) -> None:
        """
        Generate and write a starter Jupyter notebook for the project.

        This method creates a Jupyter notebook with pre-defined cells
        to help users get started with their Bukka project.
        """
        starter_notebook_writer = StarterNotebookWriter(
            output_path=str(self.file_manager.starter_notebook_path)
        )

        logger.info("Writing starter notebook")
        starter_notebook_writer.write_notebook()