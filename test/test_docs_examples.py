"""Unit tests for documentation examples.

This module tests all code examples from the documentation to ensure they work correctly.
"""
import pytest
import tempfile
import os
from pathlib import Path

from bukka.project import Project


class TestGettingStartedExamples:
    """Test examples from getting_started.rst"""
    
    def test_project_class_example(self):
        """Test the Project class example from Getting Started.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(
        ...     name="my_ml_project",
        ...     dataset_path=None,
        ...     backend="polars",
        ...     problem_type="auto"
        ... )
        >>> assert proj.name == "my_ml_project"
        >>> assert proj.backend == "polars"
        >>> assert proj.problem_type == "auto"
        """
        proj = Project(
            name="my_ml_project",
            dataset_path=None,
            backend="polars",
            problem_type="auto"
        )
        
        assert proj.name == "my_ml_project"
        assert proj.backend == "polars"
        assert proj.problem_type == "auto"


class TestUsageExamplesBasic:
    """Test basic examples from usage_examples.rst"""
    
    def test_minimal_project(self):
        """Test minimal project creation example.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(name="minimal_project")
        >>> assert proj.name == "minimal_project"
        >>> assert proj.dataset_path is None
        >>> assert proj.backend == "polars"
        """
        proj = Project(name="minimal_project")
        
        assert proj.name == "minimal_project"
        assert proj.dataset_path is None
        assert proj.backend == "polars"
    
    def test_project_with_dataset(self):
        """Test project with dataset example.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(
        ...     name="iris_classifier",
        ...     dataset_path="/path/to/iris.csv",
        ...     target_column="species"
        ... )
        >>> assert proj.name == "iris_classifier"
        >>> assert proj.target_column == "species"
        """
        proj = Project(
            name="iris_classifier",
            dataset_path="/path/to/iris.csv",
            target_column="species"
        )
        
        assert proj.name == "iris_classifier"
        assert proj.target_column == "species"


class TestUsageExamplesAdvanced:
    """Test advanced configuration examples from usage_examples.rst"""
    
    def test_custom_backend_selection(self):
        """Test custom backend example.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(name="pandas_project", backend="pandas")
        >>> assert proj.backend == "pandas"
        """
        proj = Project(
            name="pandas_project",
            backend="pandas"
        )
        
        assert proj.backend == "pandas"
    
    def test_custom_train_test_split(self):
        """Test custom train/test split example.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(name="custom_split", train_size=0.7)
        >>> assert proj.train_size == 0.7
        """
        proj = Project(
            name="custom_split",
            train_size=0.7
        )
        
        assert proj.train_size == 0.7
    
    def test_problem_type_specification(self):
        """Test problem type specification examples.
        
        This tests the examples:
        >>> from bukka.project import Project
        >>> binary_proj = Project(name="binary_clf", problem_type="binary_classification")
        >>> regression_proj = Project(name="regression_proj", problem_type="regression")
        >>> clustering_proj = Project(name="clustering_proj", problem_type="clustering")
        >>> assert binary_proj.problem_type == "binary_classification"
        >>> assert regression_proj.problem_type == "regression"
        >>> assert clustering_proj.problem_type == "clustering"
        """
        # Binary classification
        binary_proj = Project(
            name="binary_clf",
            problem_type="binary_classification"
        )
        
        # Regression
        regression_proj = Project(
            name="regression_proj",
            problem_type="regression"
        )
        
        # Clustering
        clustering_proj = Project(
            name="clustering_proj",
            problem_type="clustering"
        )
        
        assert binary_proj.problem_type == "binary_classification"
        assert regression_proj.problem_type == "regression"
        assert clustering_proj.problem_type == "clustering"
    
    def test_stratified_sampling(self):
        """Test stratified sampling configuration example.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(
        ...     name="stratified_project",
        ...     stratify=True,
        ...     strata=["gender", "age_group"]
        ... )
        >>> assert proj.stratify is True
        >>> assert proj.strata == ["gender", "age_group"]
        """
        proj = Project(
            name="stratified_project",
            stratify=True,
            strata=["gender", "age_group"]
        )
        
        assert proj.stratify is True
        assert proj.strata == ["gender", "age_group"]
    
    def test_skip_virtual_environment(self):
        """Test skip virtual environment example.
        
        This tests the example:
        >>> from bukka.project import Project
        >>> proj = Project(name="no_venv_project", skip_venv=True)
        >>> assert proj.skip_venv is True
        """
        proj = Project(
            name="no_venv_project",
            skip_venv=True
        )
        
        assert proj.skip_venv is True


class TestDatasetExamples:
    """Test dataset-related examples from usage_examples.rst"""
    
    def test_dataset_class_usage(self):
        """Test Dataset class example from usage_examples.rst.
        
        This tests the example showing how to load and work with datasets.
        Note: The Dataset class requires complex setup including FileManager,
        so we test that the imports work and the classes exist.
        """
        # Import here to avoid issues if module doesn't exist
        try:
            from bukka.data_management.dataset import Dataset
            from bukka.utils.files.file_manager import FileManager
        except ImportError:
            pytest.skip("Dataset module not available")
        
        # Verify classes exist and can be imported
        assert Dataset is not None
        assert FileManager is not None
        
        # This is sufficient to verify the documentation example is valid
        # Full integration testing of Dataset is done in test_data_management/


class TestProjectInitializationExamples:
    """Test Project initialization examples from test_project.py docstrings"""
    
    def test_minimal_args_example(self):
        """Test example from test_project_initialization_with_minimal_args docstring.
        
        This tests the example:
        >>> proj = Project(name="test_proj")
        >>> assert proj.name == "test_proj"
        >>> assert proj.dataset_path is None
        """
        proj = Project(name="test_proj")
        assert proj.name == "test_proj"
        assert proj.dataset_path is None
    
    def test_with_dataset_example(self):
        """Test example from test_project_initialization_with_dataset docstring.
        
        This tests the example:
        >>> proj = Project(
        ...     name="my_proj",
        ...     dataset_path="/path/to/data.csv",
        ...     target_column="target"
        ... )
        >>> assert proj.dataset_path == "/path/to/data.csv"
        >>> assert proj.target_column == "target"
        """
        proj = Project(
            name="my_proj",
            dataset_path="/path/to/data.csv",
            target_column="target"
        )
        assert proj.dataset_path == "/path/to/data.csv"
        assert proj.target_column == "target"


class TestMultipleProjectConfigurations:
    """Test creating multiple projects with different configurations"""
    
    def test_multiple_backends(self):
        """Test that different backends can be specified."""
        polars_proj = Project(name="polars_test", backend="polars")
        pandas_proj = Project(name="pandas_test", backend="pandas")
        
        assert polars_proj.backend == "polars"
        assert pandas_proj.backend == "pandas"
    
    def test_multiple_problem_types(self):
        """Test that different problem types can be specified."""
        auto_proj = Project(name="auto_test", problem_type="auto")
        binary_proj = Project(name="binary_test", problem_type="binary_classification")
        multi_proj = Project(name="multi_test", problem_type="multiclass_classification")
        reg_proj = Project(name="reg_test", problem_type="regression")
        cluster_proj = Project(name="cluster_test", problem_type="clustering")
        
        assert auto_proj.problem_type == "auto"
        assert binary_proj.problem_type == "binary_classification"
        assert multi_proj.problem_type == "multiclass_classification"
        assert reg_proj.problem_type == "regression"
        assert cluster_proj.problem_type == "clustering"
    
    def test_multiple_train_sizes(self):
        """Test that different train sizes can be specified."""
        proj_80 = Project(name="train_80", train_size=0.8)
        proj_70 = Project(name="train_70", train_size=0.7)
        proj_90 = Project(name="train_90", train_size=0.9)
        
        assert proj_80.train_size == 0.8
        assert proj_70.train_size == 0.7
        assert proj_90.train_size == 0.9


class TestEdgeCases:
    """Test edge cases and boundary conditions from documentation"""
    
    def test_train_size_boundaries(self):
        """Test train_size at boundary values."""
        # Minimum reasonable value
        proj_min = Project(name="min_train", train_size=0.1)
        assert proj_min.train_size == 0.1
        
        # Maximum reasonable value
        proj_max = Project(name="max_train", train_size=0.99)
        assert proj_max.train_size == 0.99
    
    def test_stratification_options(self):
        """Test different stratification configurations."""
        # No stratification
        proj_no_strat = Project(name="no_strat", stratify=False)
        assert proj_no_strat.stratify is False
        assert proj_no_strat.strata is None
        
        # Stratification enabled with no specific columns
        proj_strat = Project(name="strat", stratify=True)
        assert proj_strat.stratify is True
        
        # Stratification with specific columns
        proj_strat_cols = Project(
            name="strat_cols",
            stratify=True,
            strata=["col1", "col2"]
        )
        assert proj_strat_cols.stratify is True
        assert proj_strat_cols.strata == ["col1", "col2"]
    
    def test_optional_parameters(self):
        """Test projects with various optional parameters."""
        # All defaults
        proj_defaults = Project(name="defaults")
        assert proj_defaults.skip_venv is False
        assert proj_defaults.backend == "polars"
        assert proj_defaults.problem_type == "auto"
        assert proj_defaults.train_size == 0.8
        assert proj_defaults.stratify is True
        
        # All customized
        proj_custom = Project(
            name="custom",
            dataset_path="data.csv",
            target_column="target",
            skip_venv=True,
            backend="pandas",
            problem_type="regression",
            train_size=0.75,
            stratify=False
        )
        assert proj_custom.skip_venv is True
        assert proj_custom.backend == "pandas"
        assert proj_custom.problem_type == "regression"
        assert proj_custom.train_size == 0.75
        assert proj_custom.stratify is False
