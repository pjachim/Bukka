"""
Unit tests for JupyterWriter with venv support.
"""
import json
import tempfile
from pathlib import Path
import pytest

from bukka.coding.utils.jupyter_handler import JupyterWriter


class TestJupyterWriter:
    """Tests for JupyterWriter class."""

    def test_basic_notebook_creation(self) -> None:
        """Test creating a basic notebook without venv."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
            temp_path = f.name
        
        try:
            writer = JupyterWriter(temp_path)
            writer.add_cell("# Test Header", "markdown")
            writer.add_cell("print('hello')", "code")
            writer.write_notebook()
            
            # Read and verify the notebook
            with open(temp_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            assert len(notebook['cells']) == 2
            assert notebook['cells'][0]['cell_type'] == 'markdown'
            assert notebook['cells'][1]['cell_type'] == 'code'
            assert 'metadata' in notebook
            assert 'kernelspec' in notebook['metadata']
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_notebook_with_venv_path(self) -> None:
        """Test creating a notebook with venv path (but venv doesn't exist)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
            temp_path = f.name
        
        try:
            # Create a fake venv path that doesn't exist
            fake_venv = Path(tempfile.gettempdir()) / "nonexistent_venv"
            
            writer = JupyterWriter(temp_path, venv_path=fake_venv)
            writer.add_cell("# Test", "markdown")
            writer.write_notebook()
            
            # Read and verify the notebook
            with open(temp_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Since venv doesn't exist, vscode metadata should not be added
            assert 'vscode' not in notebook['metadata']
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_notebook_with_existing_venv(self) -> None:
        """Test creating a notebook with an existing venv path."""
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
                writer = JupyterWriter(temp_path, venv_path=venv_path)
                writer.add_cell("# Test", "markdown")
                writer.write_notebook()
                
                # Read and verify the notebook
                with open(temp_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Since venv exists, vscode metadata should be added
                assert 'vscode' in notebook['metadata']
                assert 'interpreter' in notebook['metadata']['vscode']
                assert 'path' in notebook['metadata']['language_info']
                assert str(python_exe.resolve()) == notebook['metadata']['language_info']['path']
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_context_manager(self) -> None:
        """Test that JupyterWriter works as a context manager."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
            temp_path = f.name
        
        try:
            with JupyterWriter(temp_path) as writer:
                writer.add_cell("# Test", "markdown")
            
            # Verify file was created
            assert Path(temp_path).exists()
            
            # Verify content
            with open(temp_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            assert len(notebook['cells']) == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)
