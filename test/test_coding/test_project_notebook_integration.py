"""
Integration test demonstrating venv-connected notebook generation.

This test shows the complete flow from Project creation to
notebook generation with venv configuration.
"""
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch

from bukka.project import Project


class TestProjectNotebookIntegration:
    """Integration tests for Project notebook generation with venv."""

    @patch('bukka.environment.environment.EnvironmentBuilder.build_environment')
    def test_project_creates_venv_connected_notebook(self, mock_build_env) -> None:
        """Test that Project creates a notebook connected to the venv."""
        # Create temporary dataset
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("feature1,feature2,target\n")
            f.write("1,2,0\n")
            f.write("3,4,1\n")
            dataset_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            
            try:
                # Create project WITHOUT skipping venv
                project = Project(
                    name=str(project_path),
                    dataset_path=dataset_path,
                    target_column="target",
                    skip_venv=False  # Venv should be set up
                )
                
                # Build skeleton and setup
                project._build_skeleton()
                
                # Mock the venv Python executable to exist
                venv_path = project.file_manager.virtual_env
                import sys
                if sys.platform == 'win32':
                    python_dir = venv_path / "Scripts"
                    python_exe = python_dir / "python.exe"
                else:
                    python_dir = venv_path / "bin"
                    python_exe = python_dir / "python"
                
                python_dir.mkdir(parents=True, exist_ok=True)
                python_exe.touch()
                
                # Write the notebook
                project._write_starter_notebook()
                
                # Verify the notebook was created
                notebook_path = project.file_manager.starter_notebook_path
                assert notebook_path.exists()
                
                # Read and verify venv configuration
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Should have vscode metadata with interpreter path
                assert 'vscode' in notebook['metadata']
                assert 'interpreter' in notebook['metadata']['vscode']
                assert 'path' in notebook['metadata']['language_info']
                assert str(python_exe.resolve()) == notebook['metadata']['language_info']['path']
                
            finally:
                Path(dataset_path).unlink(missing_ok=True)

    def test_project_creates_notebook_without_venv_when_skipped(self) -> None:
        """Test that Project creates a notebook without venv config when skip_venv=True."""
        # Create temporary dataset
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("feature1,feature2,target\n")
            f.write("1,2,0\n")
            f.write("3,4,1\n")
            dataset_path = f.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            
            try:
                # Create project WITH skip_venv
                project = Project(
                    name=str(project_path),
                    dataset_path=dataset_path,
                    target_column="target",
                    skip_venv=True  # Skip venv setup
                )
                
                # Build skeleton
                project._build_skeleton()
                
                # Write the notebook
                project._write_starter_notebook()
                
                # Verify the notebook was created
                notebook_path = project.file_manager.starter_notebook_path
                assert notebook_path.exists()
                
                # Read and verify NO venv configuration
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Should NOT have vscode metadata (since venv was skipped)
                assert 'vscode' not in notebook['metadata']
                
            finally:
                Path(dataset_path).unlink(missing_ok=True)
