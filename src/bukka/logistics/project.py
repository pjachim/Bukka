from bukka.logistics.files.file_manager import FileManager
from bukka.logistics.environment.environment import EnvironmentBuilder

class Project:
    """
    Represents a data science or ML project, managing its file structure and environment setup.
    """
    def __init__(self, name: str, dataset_path: str) -> None:
        """
        Initialize a Project instance.

        Args:
            name (str): The name of the project (used as the project path).
            dataset_path (str): The path to the original dataset file.
        """
        self.name: str = name
        self.dataset_path: str = dataset_path
        self.file_manager: FileManager | None = None
        self.environ_manager: EnvironmentBuilder | None = None

    def run(self) -> None:
        """
        Run the project setup: build the file skeleton and set up the environment.
        """
        self._build_skeleton()
        self._setup_environment()

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