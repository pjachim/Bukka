"""Unit tests for MLflow notebook generation."""

import json
import tempfile
from pathlib import Path

import pytest

from bukka.coding.write_mlflow_notebook import MLflowNotebookWriter
from bukka.project import Project


pytestmark = pytest.mark.venv


class TestMLflowNotebookWriter:
    """Tests for MLflow notebook content generation."""

    def test_classification_notebook_uses_project_data_and_dummy_classifier(self) -> None:
        """Classification notebooks should use DataReader and DummyClassifier."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as handle:
            notebook_path = handle.name

        try:
            writer = MLflowNotebookWriter(
                output_path=notebook_path,
                target_column="survived",
                problem_type="binary_classification",
            )
            writer.write_notebook()

            with open(notebook_path, "r", encoding="utf-8") as handle:
                notebook = json.load(handle)

            cell_contents = " ".join("".join(cell["source"]) for cell in notebook["cells"])

            assert "from utils.data_reader import DataReader" in cell_contents
            assert "DummyClassifier" in cell_contents
            assert "data_reader.readXy_train(return_pandas=True)" in cell_contents
            assert "dummy_baseline" in cell_contents
            assert 'mlflow.log_metric("test_score", test_accuracy)' in cell_contents
            assert "load_iris" not in cell_contents
            assert "RandomForestClassifier" not in cell_contents
        finally:
            Path(notebook_path).unlink(missing_ok=True)

    def test_regression_notebook_uses_dummy_regressor_metrics(self) -> None:
        """Regression notebooks should log regression-specific metrics."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as handle:
            notebook_path = handle.name

        try:
            writer = MLflowNotebookWriter(
                output_path=notebook_path,
                target_column="target",
                problem_type="regression",
            )
            writer.write_notebook()

            with open(notebook_path, "r", encoding="utf-8") as handle:
                notebook = json.load(handle)

            cell_contents = " ".join("".join(cell["source"]) for cell in notebook["cells"])

            assert "DummyRegressor" in cell_contents
            assert 'mlflow.log_metric("test_rmse", test_rmse)' in cell_contents
            assert "mean_absolute_error" in cell_contents
            assert "classification_report" not in cell_contents
        finally:
            Path(notebook_path).unlink(missing_ok=True)

    def test_notebook_with_venv_adds_interpreter_metadata(self) -> None:
        """MLflow notebooks should carry venv metadata when available."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ipynb") as handle:
            notebook_path = handle.name

        with tempfile.TemporaryDirectory() as temp_venv:
            venv_path = Path(temp_venv)

            import sys

            if sys.platform == "win32":
                python_dir = venv_path / "Scripts"
                python_exe = python_dir / "python.exe"
            else:
                python_dir = venv_path / "bin"
                python_exe = python_dir / "python"

            python_dir.mkdir(parents=True, exist_ok=True)
            python_exe.touch()

            try:
                writer = MLflowNotebookWriter(
                    output_path=notebook_path,
                    venv_path=venv_path,
                    target_column="target",
                    problem_type="regression",
                )
                writer.write_notebook()

                with open(notebook_path, "r", encoding="utf-8") as handle:
                    notebook = json.load(handle)

                assert "vscode" in notebook["metadata"]
                assert str(python_exe.resolve()) == notebook["metadata"]["language_info"]["path"]
            finally:
                Path(notebook_path).unlink(missing_ok=True)

    def test_project_writer_uses_project_problem_context(self, tmp_path: Path) -> None:
        """Project-level notebook generation should pass supervised context through."""
        dataset_path = tmp_path / "dataset.csv"
        dataset_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")

        project_path = tmp_path / "demo_project"
        project = Project(
            name=str(project_path),
            dataset_path=str(dataset_path),
            target_column="target",
            problem_type="regression",
            skip_venv=True,
            enable_mlflow=True,
        )

        project._build_skeleton()
        project._write_mlflow_notebook()

        with open(project.file_manager.mlflow_notebook_path, "r", encoding="utf-8") as handle:
            notebook = json.load(handle)

        cell_contents = " ".join("".join(cell["source"]) for cell in notebook["cells"])

        assert "DummyRegressor" in cell_contents
        assert 'mlflow.log_param("target_column", "target")' in cell_contents