"""CLI configuration management for Bukka projects.

This module handles YAML configuration files for project setup,
including validation and default values.
"""
from pathlib import Path
from typing import Any
import yaml


# Narwhals-supported backends
SUPPORTED_BACKENDS = ["polars", "pandas", "modin", "cudf", "dask", "pyarrow"]

# Problem types
PROBLEM_TYPES = [
    "binary_classification",
    "multiclass_classification", 
    "regression",
    "clustering",
    "auto"  # Let the system detect the problem type
]

DEFAULT_CONFIG = {
    "project": {
        "name": "my_bukka_project",
        "dataset": None,
        "target": None,
        "skip_venv": False,
    },
    "data": {
        "backend": "polars",
        "train_size": 0.8,
        "stratify": True,
        "strata": None,
    },
    "problem": {
        "type": "auto",  # auto-detect or specify: binary_classification, multiclass_classification, regression, clustering
    },
}


class ConfigValidator:
    """Validates Bukka configuration parameters."""

    @staticmethod
    def validate_backend(backend: str) -> str:
        """Validate the dataframe backend choice.
        
        Args:
            backend: Backend name to validate.
            
        Returns:
            The validated backend name.
            
        Raises:
            ValueError: If backend is not supported.
            
        Examples:
            >>> ConfigValidator.validate_backend("polars")
            'polars'
            >>> ConfigValidator.validate_backend("invalid")
            Traceback (most recent call last):
                ...
            ValueError: Backend 'invalid' not supported...
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Backend '{backend}' not supported. "
                f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
            )
        return backend

    @staticmethod
    def validate_problem_type(problem_type: str | None) -> str:
        """Validate the problem type specification.
        
        Args:
            problem_type: Problem type to validate.
            
        Returns:
            The validated problem type.
            
        Raises:
            ValueError: If problem type is not recognized.
            
        Examples:
            >>> ConfigValidator.validate_problem_type("regression")
            'regression'
            >>> ConfigValidator.validate_problem_type("invalid")
            Traceback (most recent call last):
                ...
            ValueError: Problem type 'invalid' not recognized...
        """
        if problem_type is None:
            return "auto"
        
        if problem_type not in PROBLEM_TYPES:
            raise ValueError(
                f"Problem type '{problem_type}' not recognized. "
                f"Supported types: {', '.join(PROBLEM_TYPES)}"
            )
        return problem_type

    @staticmethod
    def validate_dataset_path(dataset_path: str | None) -> Path | None:
        """Validate the dataset path exists.
        
        Args:
            dataset_path: Path to the dataset file.
            
        Returns:
            Validated Path object or None.
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            
        Examples:
            >>> ConfigValidator.validate_dataset_path(None)
            >>> path = ConfigValidator.validate_dataset_path("data.csv")
            >>> isinstance(path, Path)
            True
        """
        if dataset_path is None:
            return None
            
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        if not path.is_file():
            raise ValueError(f"Dataset path is not a file: {dataset_path}")
            
        return path

    @staticmethod
    def validate_train_size(train_size: float) -> float:
        """Validate the train/test split ratio.
        
        Args:
            train_size: Proportion of data for training (0 < train_size < 1).
            
        Returns:
            The validated train size.
            
        Raises:
            ValueError: If train_size is not between 0 and 1.
            
        Examples:
            >>> ConfigValidator.validate_train_size(0.8)
            0.8
            >>> ConfigValidator.validate_train_size(1.5)
            Traceback (most recent call last):
                ...
            ValueError: train_size must be between 0 and 1...
        """
        if not 0 < train_size < 1:
            raise ValueError(
                f"train_size must be between 0 and 1, got {train_size}"
            )
        return train_size

    @staticmethod
    def validate_project_name(name: str) -> str:
        """Validate the project name.
        
        Args:
            name: Project name to validate.
            
        Returns:
            The validated project name.
            
        Raises:
            ValueError: If name is empty or contains invalid characters.
            
        Examples:
            >>> ConfigValidator.validate_project_name("my_project")
            'my_project'
            >>> ConfigValidator.validate_project_name("")
            Traceback (most recent call last):
                ...
            ValueError: Project name cannot be empty
        """
        if not name or not name.strip():
            raise ValueError("Project name cannot be empty")
        
        # Check for potentially problematic characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in name for char in invalid_chars):
            raise ValueError(
                f"Project name contains invalid characters. "
                f"Avoid: {', '.join(invalid_chars)}"
            )
            
        return name.strip()


class ConfigManager:
    """Manages YAML configuration files for Bukka projects."""

    @staticmethod
    def create_template(output_path: str | Path = "bukka_config.yaml") -> Path:
        """Create a YAML configuration template with default values.
        
        Args:
            output_path: Path where the template should be written.
            
        Returns:
            Path to the created template file.
            
        Examples:
            >>> path = ConfigManager.create_template("config.yaml")
            >>> path.exists()
            True
        """
        output_path = Path(output_path)
        
        # Add comments to the YAML for better user experience
        yaml_content = f"""# Bukka Project Configuration Template
# This file contains all available configuration options for a Bukka project.

# Project settings
project:
  name: {DEFAULT_CONFIG['project']['name']}  # Required: Name of your project
  dataset: null  # Path to your dataset file (CSV, Parquet, etc.)
  target: null  # Name of the target column in your dataset
  skip_venv: {DEFAULT_CONFIG['project']['skip_venv']}  # Skip virtual environment creation

# Data processing settings
data:
  backend: {DEFAULT_CONFIG['data']['backend']}  # Dataframe backend: {', '.join(SUPPORTED_BACKENDS)}
  train_size: {DEFAULT_CONFIG['data']['train_size']}  # Proportion of data for training (0.0 - 1.0)
  stratify: {DEFAULT_CONFIG['data']['stratify']}  # Whether to stratify the train/test split
  strata: null  # Column(s) to stratify on (list or single column name)

# Problem specification
problem:
  type: {DEFAULT_CONFIG['problem']['type']}  # Problem type: {', '.join(PROBLEM_TYPES)}
"""
        
        output_path.write_text(yaml_content, encoding='utf-8')
        return output_path

    @staticmethod
    def load_config(config_path: str | Path) -> dict[str, Any]:
        """Load and validate configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Dictionary containing validated configuration.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
            
        Examples:
            >>> # Assuming valid config file exists
            >>> config = ConfigManager.load_config("config.yaml")
            >>> 'project' in config
            True
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")
        
        # Merge with defaults
        merged_config = ConfigManager._merge_with_defaults(config)
        
        # Validate the configuration
        ConfigManager._validate_config(merged_config)
        
        return merged_config

    @staticmethod
    def _merge_with_defaults(config: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with default values.
        
        Args:
            config: User-provided configuration.
            
        Returns:
            Configuration merged with defaults.
        """
        merged = DEFAULT_CONFIG.copy()
        
        for section in ['project', 'data', 'problem']:
            if section in config and isinstance(config[section], dict):
                merged[section].update(config[section])
        
        return merged

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        """Validate configuration values.
        
        Args:
            config: Configuration to validate.
            
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate project name
        if config['project']['name']:
            ConfigValidator.validate_project_name(config['project']['name'])
        
        # Validate backend
        ConfigValidator.validate_backend(config['data']['backend'])
        
        # Validate problem type
        ConfigValidator.validate_problem_type(config['problem']['type'])
        
        # Validate train size
        ConfigValidator.validate_train_size(config['data']['train_size'])
        
        # Validate dataset path if provided
        if config['project']['dataset']:
            ConfigValidator.validate_dataset_path(config['project']['dataset'])
