from pathlib import Path
import shutil, sys
from typing import Union, List

# Define a type alias for paths that can be represented as strings or Path objects.
PathLike = Union[str, Path]

class FileManager:
    """
    Manages the creation and organization of a standardized project directory structure.

    This class handles path construction for various project components (data, pipelines,
    scripts, virtual environment) and provides a method to build the directory skeleton
    and copy the initial dataset.

    Attributes
    ----------
    project_path : Path
        The root directory of the project.
    orig_dataset : Path
        The path to the original dataset file provided by the user.
    data_path : Path
        Path to the main 'data' directory (project_path / 'data').
    train_data : Path
        Path to the 'train' data subdirectory (data_path / 'train').
    test_data : Path
        Path to the 'test' data subdirectory (data_path / 'test').
    pipes : Path
        Path to the main 'pipelines' directory (project_path / 'pipelines').
    generated_pipes : Path
        Path to the 'generated' pipelines subdirectory.
    baseline_pipes : Path
        Path to the 'baseline' pipelines subdirectory.
    candidate_pipes : Path
        Path to the 'candidate' pipelines subdirectory.
    virtual_env : Path
        Path to the project's virtual environment directory (project_path / '.venv').
    scripts : Path
        Path to the 'scripts' directory.
    dataset_path : Path
        The final destination path for the copied dataset (data_path / dataset_filename).
    starter_notebook_path : Path
        Path for the 'starter.ipynb' file.
    requirements_path : Path
        Path for the 'requirements.txt' file.
    readme_path : Path
        Path for the 'README.md' file.
    gitignore_path : Path
        Path for the '.gitignore' file.
    """
    def __init__(
        self,
        project_path: PathLike,
        orig_dataset: PathLike
    ) -> None:
        """
        Initializes the FileManager with project and dataset paths.

        Parameters
        ----------
        project_path : PathLike
            The path to the project's root directory. Can be a string or a Path object.
        orig_dataset : PathLike
            The path to the original dataset file to be copied. Can be a string or a Path object.
        """
        # Convert all input path arguments to Path objects for consistent handling.
        self.project_path: Path = Path(project_path)

        if orig_dataset is not None:
            self.orig_dataset: Path = Path(orig_dataset)
        else:
            self.orig_dataset = None

        # Build all necessary directory and file paths immediately upon initialization.
        self._build_paths()

    # --- Internal Path Management Methods ---

    def _build_paths(self) -> None:
        """
        Constructs all standard paths required for the project structure.

        This method populates the object's attributes with Path objects representing
        all intended directories and key files. This is called automatically in __init__.
        """
        # Data related paths
        self.data_path: Path = self.project_path / 'data'
        self.train_data: Path = self.data_path / 'train'
        self.test_data: Path = self.data_path / 'test'

        self.train_data_file: Path = self.train_data / 'train_data.pqt'
        self.test_data_file: Path = self.test_data / 'test_data.pqt'

        # Pipeline related paths (treated as a Python package)
        self.pipes: Path = self.project_path / 'pipelines'
        self.generated_pipes: Path = self.pipes / 'generated'
        self.baseline_pipes: Path = self.pipes / 'baseline'
        self.candidate_pipes: Path = self.pipes / 'candidate'

        # Other top-level paths
        self.virtual_env: Path = self.project_path / '.venv'
        # Utils path (treated as a Python package)
        self.utils: Path = self.project_path / 'utils'
        self.data_reader_path: Path = self.utils / 'data_reader.py'

        # Determine the final dataset path within the 'data' directory.
        # It uses the original file's name.
        if self.orig_dataset is not None:
            self.dataset_path: Path = self.data_path / self.orig_dataset.name

        # Paths for standard files (these will not be created by build_skeleton
        # but are tracked for consistency/future methods)
        self.starter_notebook_path: Path = self.project_path / 'starter.ipynb'
        self.requirements_path: Path = self.project_path / 'requirements.txt'
        self.readme_path: Path = self.project_path / 'README.md'
        self.gitignore_path: Path = self.project_path / '.gitignore'

    def _make_path(self, path: Path) -> None:
        """
        Creates a directory if it doesn't already exist.

        Parameters
        ----------
        path : Path
            The Path object representing the directory to create.
        """
        # parents=True ensures any necessary parent directories are also created.
        # exist_ok=True prevents an error if the directory already exists.
        path.mkdir(parents=True, exist_ok=True)

    # --- Public Interface Methods ---

    def build_skeleton(self) -> None:
        """
        Creates the defined project directory structure and copies the dataset.

        It systematically creates all necessary folders and adds __init__.py files
        to designated directories to treat them as Python packages. Finally, it
        copies the original dataset into the new data folder.
        """
        # --- 1. Create Data Directories ---
        self._make_path(self.data_path)
        self._make_path(self.train_data)
        self._make_path(self.test_data)

        # --- 2. Create Pipeline Directories (with __init__.py) ---
        self._make_path(self.pipes)
        # Create __init__.py to make 'pipelines' a package
        (self.pipes / '__init__.py').touch()

        self._make_path(self.generated_pipes)
        # Create __init__.py to make 'generated' a subpackage
        (self.generated_pipes / '__init__.py').touch()

        self._make_path(self.baseline_pipes)
        # Create __init__.py to make 'baseline' a subpackage
        (self.baseline_pipes / '__init__.py').touch()

        self._make_path(self.candidate_pipes)
        # Create __init__.py to make 'candidate' a subpackage
        (self.candidate_pipes / '__init__.py').touch()

        # --- 3. Create Other Directories (Virtual Env and Utils) ---
        self._make_path(self.virtual_env)

        self._make_path(self.utils)
        # Create __init__.py to make 'utils' a package
        (self.utils / '__init__.py').touch()

        # --- 4. Copy Dataset ---
        # The copy2 function is used as it attempts to preserve metadata (like timestamps).
        # This copies the source file (orig_dataset) to the destination path (dataset_path).
        if self.orig_dataset is not None:
            try:
                shutil.copy2(self.orig_dataset, self.dataset_path)
            except FileNotFoundError as e:
                # Re-raise the error with a more informative message.
                raise FileNotFoundError(
                    f"Original dataset file not found: {self.orig_dataset}"
                ) from e
            
        if sys.platform == 'win32':
            self.python_path = self.virtual_env / "Scripts" / "python.exe"
        else:
            self.python_path = self.virtual_env / "bin" / "python"