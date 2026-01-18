"""
Unit tests for DataReaderWriter.
"""
import tempfile
from pathlib import Path
import pytest

from bukka.coding.write_data_reader_class import DataReaderWriter
from bukka.utils.files.file_manager import FileManager


class TestDataReaderWriter:
    """Tests for DataReaderWriter class."""

    def test_generates_relative_paths(self) -> None:
        """Test that DataReaderWriter uses config constants for paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            
            # Create a minimal dataset file
            dataset_file = Path(temp_dir) / "test_data.csv"
            dataset_file.write_text("col1,col2\n1,2\n")
            
            # Create file manager
            fm = FileManager(
                project_path=str(project_path),
                orig_dataset=str(dataset_file)
            )
            fm.build_skeleton()
            
            # Create DataReaderWriter and generate code
            writer = DataReaderWriter(fm)
            code = writer._fill_template()
            
            # Check that paths use config constants
            assert "TRAIN_DATASET_PATH" in code
            assert "TEST_DATASET_PATH" in code
            assert "from config import" in code
            
            # Check that absolute paths are NOT in the code
            assert str(project_path) not in code

    def test_writes_valid_python_class(self) -> None:
        """Test that the written DataReader class is valid Python."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            
            # Create a minimal dataset file
            dataset_file = Path(temp_dir) / "test_data.csv"
            dataset_file.write_text("col1,col2\n1,2\n")
            
            # Create file manager
            fm = FileManager(
                project_path=str(project_path),
                orig_dataset=str(dataset_file)
            )
            fm.build_skeleton()
            
            # Write the DataReader class
            writer = DataReaderWriter(fm)
            writer.write_code()
            
            # Verify the file was created
            assert fm.data_reader_path.exists()
            
            # Read and verify it's valid Python
            code = fm.data_reader_path.read_text()
            
            # Should be able to compile it
            compile(code, str(fm.data_reader_path), 'exec')
            
            # Check for expected class structure
            assert "class DataReader:" in code
            assert "def read_train_data" in code
            assert "def read_test_data" in code
