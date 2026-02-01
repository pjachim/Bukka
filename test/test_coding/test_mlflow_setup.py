"""Unit tests for MLflow setup file generation."""
import pytest
from pathlib import Path
from bukka.coding.write_mlflow_setup import MLflowSetupWriter
from bukka.utils.files.file_manager import FileManager


class TestMLflowSetupWriter:
    """Test cases for MLflowSetupWriter class."""
    
    def test_mlflow_setup_writer_initialization(self, tmp_path):
        """Test MLflowSetupWriter initializes correctly."""
        # Create a temporary FileManager
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        file_manager = FileManager(project_path=project_path, orig_dataset=None)
        
        # Create writer
        writer = MLflowSetupWriter(
            file_manager=file_manager,
            project_name="test_project"
        )
        
        # Verify output_path is set correctly
        assert writer.output_path == file_manager.mlflow_setup_path
    
    def test_mlflow_setup_writer_with_custom_uri(self, tmp_path):
        """Test MLflowSetupWriter with custom tracking URI."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        file_manager = FileManager(project_path=project_path, orig_dataset=None)
        
        custom_uri = "http://localhost:5000"
        writer = MLflowSetupWriter(
            file_manager=file_manager,
            project_name="test_project",
            tracking_uri=custom_uri
        )
        
        # Verify output_path is set correctly
        assert writer.output_path == file_manager.mlflow_setup_path
    
    def test_mlflow_setup_writer_creates_file(self, tmp_path):
        """Test that MLflowSetupWriter creates the setup file."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        # Ensure scripts directory exists
        scripts_path = project_path / "scripts"
        scripts_path.mkdir(exist_ok=True)
        
        file_manager = FileManager(project_path=project_path, orig_dataset=None)
        
        writer = MLflowSetupWriter(
            file_manager=file_manager,
            project_name="test_project"
        )
        writer.write_code()
        
        # Verify file was created
        assert file_manager.mlflow_setup_path.exists()
        
        # Verify content
        content = file_manager.mlflow_setup_path.read_text()
        assert "import mlflow" in content
        assert "def setup_mlflow()" in content
        assert "from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME" in content
        assert "setup_mlflow" in content
    
    def test_mlflow_setup_template_has_required_functions(self, tmp_path):
        """Test that generated MLflow setup has required functions."""
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        # Ensure scripts directory exists
        scripts_path = project_path / "scripts"
        scripts_path.mkdir(exist_ok=True)
        
        file_manager = FileManager(project_path=project_path, orig_dataset=None)
        
        writer = MLflowSetupWriter(
            file_manager=file_manager,
            project_name="my_experiment"
        )
        writer.write_code()
        
        content = file_manager.mlflow_setup_path.read_text()
        
        # Check for setup function
        assert "def setup_mlflow()" in content
        assert "mlflow.set_tracking_uri" in content
        assert "mlflow.set_experiment" in content
        assert "from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME" in content
        
        # Check for main block
        assert 'if __name__ == "__main__"' in content
