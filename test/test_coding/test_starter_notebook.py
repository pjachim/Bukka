"""
Unit tests for StarterNotebookWriter with venv support.
"""
import json
import tempfile
from pathlib import Path
import pytest

from bukka.coding.write_starter_notebook import StarterNotebookWriter


class TestStarterNotebookWriter:
    """Tests for StarterNotebookWriter class."""

    def test_create_notebook_without_venv(self) -> None:
        """Test creating a starter notebook without venv configuration."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
            temp_path = f.name
        
        try:
            writer = StarterNotebookWriter(output_path=temp_path)
            writer.write_notebook()
            
            # Read and verify the notebook
            with open(temp_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Check that cells were created
            assert len(notebook['cells']) > 0
            
            # Check basic metadata
            assert 'metadata' in notebook
            assert 'kernelspec' in notebook['metadata']
            
            # vscode metadata should not be present without venv
            assert 'vscode' not in notebook['metadata']
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_notebook_with_venv(self) -> None:
        """Test creating a starter notebook with venv configuration."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
            temp_path = f.name
        
        # Create a temporary directory structure mimicking a venv
        with tempfile.TemporaryDirectory() as temp_venv:
            venv_path = Path(temp_venv)
            
            # Create the Scripts/bin directory with python executable
            import sys
            if sys.platform == 'win32':
                python_dir = venv_path / "Scripts"
                python_exe = python_dir / "python.exe"
            else:
                python_dir = venv_path / "bin"
                python_exe = python_dir / "python"
            
            python_dir.mkdir(parents=True, exist_ok=True)
            python_exe.touch()  # Create empty file
            
            try:
                writer = StarterNotebookWriter(
                    output_path=temp_path,
                    venv_path=venv_path
                )
                writer.write_notebook()
                
                # Read and verify the notebook
                with open(temp_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Check that cells were created
                assert len(notebook['cells']) > 0
                
                # Check venv metadata
                assert 'vscode' in notebook['metadata']
                assert 'interpreter' in notebook['metadata']['vscode']
                assert 'path' in notebook['metadata']['language_info']
                assert str(python_exe.resolve()) == notebook['metadata']['language_info']['path']
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_notebook_cells_content(self) -> None:
        """Test that the starter notebook contains expected cells."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
            temp_path = f.name
        
        try:
            writer = StarterNotebookWriter(output_path=temp_path)
            writer.write_notebook()
            
            # Read and verify the notebook
            with open(temp_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            cells = notebook['cells']
            
            # Check for expected sections
            cell_contents = ' '.join([''.join(cell['source']) for cell in cells])
            
            assert 'Welcome to Your Bukka Project' in cell_contents
            assert 'Data Loading' in cell_contents
            assert 'Running a Pipeline' in cell_contents
            assert 'Model Evaluation' in cell_contents
            assert 'DataReader' in cell_contents
            assert 'pipeline' in cell_contents
        finally:
            Path(temp_path).unlink(missing_ok=True)
