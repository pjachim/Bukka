"""Unit tests for the current documentation examples.

These tests track the examples in the reStructuredText docs and keep them
aligned with the public API that the app currently exposes.
"""

from pathlib import Path

from bukka.cli_config import BukkaConfig, ConfigValidator, PROBLEM_TYPES
from bukka.project import Project


DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs" / "source"


class TestGettingStartedExamples:
    """Test the programmatic example shown in getting_started.rst."""

    def test_project_class_example(self) -> None:
        """The documented Project example still matches the constructor."""
        proj = Project(
            name="my_ml_project",
            dataset_path=None,
            backend="polars",
            problem_type="auto",
        )

        assert proj.name == "my_ml_project"
        assert proj.dataset_path is None
        assert proj.backend == "polars"
        assert proj.problem_type == "auto"


class TestUsageExamples:
    """Test the CLI examples shown in usage_examples.rst."""

    def test_usage_examples_match_current_cli_commands(self) -> None:
        """The documented commands should still appear in the docs page."""
        content = (DOCS_ROOT / "usage_examples.rst").read_text(encoding="utf-8")

        expected_snippets = [
            "python -m bukka run --name iris_classifier --dataset iris.csv --target species",
            "python -m bukka init-config",
            "python -m bukka run --config bukka_config.yaml",
            "python -m bukka run -n fraud_detection -d transactions.csv -t is_fraud",
            "--backend pandas --problem-type binary_classification",
            "--problem-type regression --train-size 0.75",
            "--problem-type clustering --backend polars",
            "python -m bukka run -n my_project -d data.csv --skip-venv",
            "python -m bukka run -n my_project -d data.csv --strata gender age_group",
            "python -m bukka run -n my_project -d data.csv --mlflow",
            "python -m bukka run -n my_project -d data.csv --dummy",
            "python -m bukka run -n my_project -d data.csv --tpot",
        ]

        for snippet in expected_snippets:
            assert snippet in content


class TestConfigurationExamples:
    """Test the current configuration surface used by the docs."""

    def test_config_defaults_match_current_project_config(self) -> None:
        """The default project config still reflects the current API."""
        config = BukkaConfig(name="my_project")

        assert config.backend == "pyarrow"
        assert config.problem_type == "auto"
        assert config.train_size == 0.8
        assert config.stratify is True
        assert config.skip_venv is False

    def test_config_validator_supports_current_problem_types(self) -> None:
        """The validator should accept the problem types exposed in code."""
        assert ConfigValidator.validate_problem_type(None) == "auto"
        assert ConfigValidator.validate_problem_type("binary_classification") == "binary_classification"
        assert ConfigValidator.validate_problem_type("multiclass_classification") == "multiclass_classification"
        assert ConfigValidator.validate_problem_type("regression") == "regression"
        assert ConfigValidator.validate_problem_type("other") == "other"

        assert PROBLEM_TYPES == [
            "binary_classification",
            "multiclass_classification",
            "regression",
            "other",
        ]

    def test_configuration_docs_still_reference_init_config(self) -> None:
        """The configuration docs should still point users at init-config."""
        content = (DOCS_ROOT / "configuration.rst").read_text(encoding="utf-8")

        assert "python -m bukka init-config" in content
        assert "python -m bukka run --config config.yaml --backend pandas" in content
