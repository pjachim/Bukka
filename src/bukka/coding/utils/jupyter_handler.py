import json
from pathlib import Path

class JupyterWriter:
    def __init__(self, filename: str, venv_path: str | Path | None = None) -> None:
        self.cells: list[dict[str, None | str | list[str]]] = []
        self.filename = filename
        self.venv_path = Path(venv_path) if venv_path else None

    def add_cell(self, cell_content: str, cell_type: str = "code") -> None:
        '''
        Write a cell to the Jupyter notebook.
        
        Args:
            cell_content: The content of the cell.
            cell_type: The type of the cell, either 'code' or 'markdown'.
        '''
        self.cells.append(self._format_cell(cell_content, cell_type))

    def write_notebook(self) -> None:
        '''Write the Jupyter notebook to file.'''
        notebook_content = self._format_notebook()
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=4)

    def _format_cell(self, cell_content: str, cell_type: str) -> dict:
        '''Format a single cell for Jupyter notebook structure.'''
        return {
            "cell_type": cell_type,
            "metadata": {},
            "source": cell_content.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None,
        }

    def _format_notebook(self) -> dict:
        metadata = {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python",
                "version": "3.x"
            }
        }
        
        # If venv_path is provided, add Python interpreter path to metadata
        if self.venv_path:
            import sys
            if sys.platform == 'win32':
                python_path = self.venv_path / "Scripts" / "python.exe"
            else:
                python_path = self.venv_path / "bin" / "python"
            
            # Only add if the Python executable exists
            if python_path.exists():
                metadata["vscode"] = {
                    "interpreter": {
                        "hash": str(hash(str(python_path.resolve()))),
                    }
                }
                metadata["language_info"]["path"] = str(python_path.resolve())
        
        notebook_content = {
            "cells": self.cells,
            "metadata": metadata,
            "nbformat": 4,
            "nbformat_minor": 2
        }

        return notebook_content
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.write_notebook()
        return False  # Do not suppress exceptions
    
    def __repr__(self):
        return f"JupyterWriter(filename={self.filename}, cells={len(self.cells)})"