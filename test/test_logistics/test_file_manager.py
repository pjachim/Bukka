import pytest
from pathlib import Path
import shutil
from typing import Generator
from logistics.files.file_manager import FileManager

# A fixture to create a temporary, isolated directory structure for testing.
# This ensures tests are clean and don't affect the actual filesystem.
@pytest.fixture
def tmp_project_setup(tmp_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """
    Fixture to create temporary project and a mock dataset file.

    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in fixture for temporary directories.

    Yields
    ------
    tuple[Path, Path]
        The path to the temporary project root and the path to the mock dataset.
    """
    # 1. Setup: Create a temporary project directory and a mock dataset file
    project_root = tmp_path / "test_project"
    project_root.mkdir()

    # Create a mock dataset file outside the project root
    mock_dataset_name = "test_data.csv"
    mock_dataset_path = tmp_path / mock_dataset_name
    mock_dataset_path.touch()
    
    # Write some content to the mock dataset to test file size/content integrity
    with open(mock_dataset_path, "w") as f:
        f.write("A,B\n1,2")

    yield project_root, mock_dataset_path

    # 2. Teardown: Cleanup is implicitly handled by tmp_path, which deletes the directory
    # and all its contents when the fixture finishes.


class TestFileManagerPaths:
    """
    Tests the path construction logic within the FileManager class.
    """

    def test_initialization_with_paths(self, tmp_project_setup: tuple[Path, Path]):
        """Test that initialization correctly converts string inputs to Path objects."""
        project_root, mock_dataset_path = tmp_project_setup
        
        # Initialize with string inputs
        fm = FileManager(str(project_root), str(mock_dataset_path))
        
        # Check that attributes are Path objects
        assert isinstance(fm.project_path, Path)
        assert isinstance(fm.orig_dataset, Path)
        
        # Check values
        assert fm.project_path == project_root
        assert fm.orig_dataset == mock_dataset_path

    def test_initialization_with_pathlikes(self, tmp_project_setup: tuple[Path, Path]):
        """Test that initialization correctly handles Path objects as inputs."""
        project_root, mock_dataset_path = tmp_project_setup
        
        # Initialize with Path objects
        fm = FileManager(project_root, mock_dataset_path)
        
        # Check values
        assert fm.project_path == project_root
        assert fm.orig_dataset == mock_dataset_path

    def test_built_path_integrity(self, tmp_project_setup: tuple[Path, Path]):
        """
        Test that all internal paths are correctly constructed relative to the project root.
        """
        project_root, mock_dataset_path = tmp_project_setup
        fm = FileManager(project_root, mock_dataset_path)

        # Expected data paths
        assert fm.data_path == project_root / 'data'
        assert fm.train_data == project_root / 'data' / 'train'
        assert fm.test_data == project_root / 'data' / 'test'
        
        # Expected dataset path within the new structure
        assert fm.dataset_path == project_root / 'data' / mock_dataset_path.name
        
        # Expected pipeline paths
        assert fm.pipes == project_root / 'pipelines'
        assert fm.generated_pipes == project_root / 'pipelines' / 'generated'
        assert fm.baseline_pipes == project_root / 'pipelines' / 'baseline'
        assert fm.candidate_pipes == project_root / 'pipelines' / 'candidate'

        # Expected miscellaneous paths
        assert fm.virtual_env == project_root / '.venv'
        assert fm.scripts == project_root / 'scripts'

        # Expected file paths (just for tracking)
        assert fm.requirements_path == project_root / 'requirements.txt'

    def test_dataset_name_extraction(self, tmp_path: Path):
        """Test that the dataset_path uses the correct file name, even with complex paths."""
        project_root = tmp_path / "proj"
        project_root.mkdir()
        
        # Create a mock dataset with a different name/location
        complex_orig_dataset = tmp_path / 'level1' / 'level2' / 'data_2025_final.dat'
        complex_orig_dataset.parent.mkdir(parents=True)
        complex_orig_dataset.touch()

        fm = FileManager(project_root, complex_orig_dataset)
        
        # Should only contain the file name, not the full source path
        assert fm.dataset_path == project_root / 'data' / 'data_2025_final.dat'


class TestFileManagerMakePath:
    """
    Tests the internal _make_path method for directory creation.
    """

    def test_make_new_single_directory(self, tmp_path: Path):
        """Test creating a simple, non-existent directory."""
        # Use a dummy FileManager instance just to access the _make_path method
        dummy_fm = FileManager(tmp_path, tmp_path / 'dummy.txt') 
        
        new_dir = tmp_path / 'new_folder'
        assert not new_dir.exists()
        
        dummy_fm._make_path(new_dir)
        
        assert new_dir.is_dir()

    def test_make_new_nested_directories(self, tmp_path: Path):
        """Test creating nested directories using parents=True."""
        dummy_fm = FileManager(tmp_path, tmp_path / 'dummy.txt')
        
        nested_dir = tmp_path / 'a' / 'b' / 'c'
        assert not nested_dir.exists()
        
        dummy_fm._make_path(nested_dir)
        
        # Check that the deepest directory exists
        assert nested_dir.is_dir()
        # Check that a parent directory was also created
        assert (tmp_path / 'a' / 'b').is_dir()

    def test_make_existing_directory(self, tmp_path: Path):
        """Test calling _make_path on an already existing directory (exist_ok=True)."""
        existing_dir = tmp_path / 'existing_folder'
        existing_dir.mkdir()
        
        # Ensure no exception is raised when the directory already exists
        dummy_fm = FileManager(tmp_path, tmp_path / 'dummy.txt')
        try:
            dummy_fm._make_path(existing_dir)
        except Exception as e:
            pytest.fail(f"_make_path failed on existing directory: {e}")
            
        assert existing_dir.is_dir() # Should still be a directory


class TestFileManagerBuildSkeleton:
    """
    Tests the main public method to build the entire project structure.
    """

    def test_skeleton_creation_success(self, tmp_project_setup: tuple[Path, Path]):
        """Test that build_skeleton creates all necessary directories and files."""
        project_root, mock_dataset_path = tmp_project_setup
        fm = FileManager(project_root, mock_dataset_path)

        fm.build_skeleton()

        # 1. Check Data Directories
        assert fm.data_path.is_dir()
        assert fm.train_data.is_dir()
        assert fm.test_data.is_dir()

        # 2. Check Pipeline Directories and __init__.py files
        pipes_dirs = [
            fm.pipes, fm.generated_pipes, fm.baseline_pipes, fm.candidate_pipes
        ]
        for pipe_dir in pipes_dirs:
            assert pipe_dir.is_dir()
            assert (pipe_dir / '__init__.py').is_file()

        # 3. Check Other Directories and __init__.py
        assert fm.virtual_env.is_dir()
        assert fm.scripts.is_dir()
        assert (fm.scripts / '__init__.py').is_file()

        # 4. Check Dataset Copy
        final_dataset_path = fm.dataset_path
        assert final_dataset_path.is_file()
        
        # Check content integrity (copied from fixture setup)
        with open(final_dataset_path, "r") as f:
            content = f.read()
        assert content == "A,B\n1,2"

    def test_skeleton_idempotence(self, tmp_project_setup: tuple[Path, Path]):
        """Test that calling build_skeleton multiple times works without error (idempotent)."""
        project_root, mock_dataset_path = tmp_project_setup
        fm = FileManager(project_root, mock_dataset_path)

        # First call: creates everything
        fm.build_skeleton()
        
        # Check initial state
        assert fm.data_path.is_dir()
        
        # Second call: should not raise an error due to exist_ok=True
        try:
            fm.build_skeleton()
        except Exception as e:
            pytest.fail(f"build_skeleton failed on second call: {e}")
            
        # Ensure the final state is still correct
        assert fm.pipes.is_dir()

    def test_skeleton_file_not_found(self, tmp_project_setup: tuple[Path, Path]):
        """Test the edge case where the original dataset file does not exist."""
        project_root, mock_dataset_path = tmp_project_setup
        
        # Delete the mock dataset file created by the fixture
        mock_dataset_path.unlink() 
        
        fm = FileManager(project_root, mock_dataset_path)

        # Expect a FileNotFoundError when shutil.copy2 is called
        with pytest.raises(FileNotFoundError) as excinfo:
            fm.build_skeleton()

        # Check for the informative error message added in the enhancement
        assert "Original dataset file not found" in str(excinfo.value)
        
        # Ensure directories were still created, even though the copy failed
        assert fm.data_path.is_dir()
        assert fm.pipes.is_dir()