from datetime import datetime
from pathlib import Path
from typing import Optional

from bukka.logistics.files.file_manager import FileManager
from bukka.logistics.environment.environment import EnvironmentBuilder
from bukka.data_management.dataset import Dataset
from bukka.expert_system.problem_identifier import ProblemIdentifier
from bukka.coding.write_pipeline import PipelineWriter

class Project:
    """
    Represents a data science or ML project, managing its file structure and environment setup.
    """
    def __init__(self, name: str, dataset_path: str, target_column: str) -> None:
        """
        Initialize a Project instance.

        Args:
            name (str): The name of the project (used as the project path).
            dataset_path (str): The path to the original dataset file.
        """
        self.name: str = name
        self.dataset_path: str = dataset_path
        self.file_manager: FileManager | None = None
        self.target_column: str = target_column
        self.environ_manager: EnvironmentBuilder | None = None

    def run(self) -> None:
        """
        Run the project setup: build the file skeleton and set up the environment.
        """
        self._build_skeleton()
        self._setup_environment()

        if self.dataset_path:
            self.write_pipeline(target_column=self.target_column)

    def write_pipeline(self, target_column: str, dataframe_backend: str = "polars") -> str:
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
        if self.file_manager is None:
            # Ensure skeleton exists and dataset is copied
            self._build_skeleton()

        dataset = Dataset(target_column, self.file_manager, dataframe_backend)
        identifier = ProblemIdentifier(dataset, target_column)
        # Run detection phases
        identifier.multivariate_problems()
        identifier.univariate_problems()
        # identify ml problem (may be clustering/regression/classification)
        try:
            identifier._identify_ml_problem()
        except Exception:
            # If private method naming changes, ignore to avoid crashing here
            pass

        # Generate pipeline
        writer = PipelineWriter(identifier)
        _, _ = writer.write()
        pipeline_text = writer.pipeline_definition or ""

        # Prepare destination file
        gen_dir: Path = self.file_manager.generated_pipes
        gen_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"pipeline_{timestamp}.py"
        dest = gen_dir / filename

        # Write pipeline text
        dest.write_text(pipeline_text, encoding="utf-8")

        return str(dest.resolve())

    def _build_skeleton(self) -> None:
        """
        Build the project file skeleton using FileManager.
        """
        self.file_manager = FileManager(
            project_path=self.name,
            orig_dataset=self.dataset_path
        )
        self.file_manager.build_skeleton()

    def _setup_environment(self) -> None:
        """
        Set up the project environment using EnvironmentBuilder.
        """
        if self.file_manager is None:
            raise RuntimeError("FileManager must be initialized before setting up the environment.")
        self.environ_manager = EnvironmentBuilder(
            file_manager=self.file_manager
        )
        self.environ_manager.build_environment()