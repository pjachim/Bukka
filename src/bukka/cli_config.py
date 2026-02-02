"""CLI configuration management for Bukka projects.

This module handles YAML configuration files for project setup,
including validation and default values.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import argparse
import yaml


# Narwhals-supported backends
SUPPORTED_BACKENDS = [
    'polars',
    'pandas',
    'modin',
    'cudf',
    'pyarrow',
    'dask',
]

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
        "enable_mlflow": False,
        "mlflow_tracking_uri": None,
    },
    "data": {
        "backend": "pyarrow",
        "train_size": 0.8,
        "stratify": True,
        "strata": None,
    },
    "problem": {
        "type": "auto",  # auto-detect or specify: binary_classification, multiclass_classification, regression, clustering
    },
}


@dataclass
class BukkaConfig:
    """Unified configuration for Bukka projects.
    
    This dataclass consolidates all project configuration parameters,
    providing a single source of truth for defaults and validation.
    
    Attributes:
        name: Project name / directory to create.
        dataset: Path to dataset file (CSV, Parquet, etc.).
        target: Name of the target column (None for clustering).
        skip_venv: Skip virtual environment creation.
        backend: Dataframe backend to use.
        train_size: Train/test split ratio (0 < x < 1).
        stratify: Whether to stratify the train/test split.
        strata: Column(s) to use for stratification.
        problem_type: ML problem type specification.
    
    Examples:
        >>> config = BukkaConfig(name="my_project")
        >>> config.backend
        'pyarrow'
        >>> config.validate()  # Returns list of errors, empty if valid
        []
    """
    # Project settings
    name: str
    dataset: str | None = None
    target: str | None = None
    skip_venv: bool = False
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    
    # Data settings
    backend: str = "pyarrow"
    train_size: float = 0.8
    stratify: bool = True
    strata: list[str] | None = None
    
    # Problem settings
    problem_type: str = "auto"
    
    @classmethod
    def from_args_and_config(
        cls, 
        args: argparse.Namespace, 
        config_dict: dict[str, Any] | None = None
    ) -> 'BukkaConfig':
        """Create BukkaConfig from CLI args and optional config file.
        
        CLI arguments take precedence over config file values.
        
        Args:
            args: Parsed command-line arguments.
            config_dict: Optional loaded config dictionary.
            
        Returns:
            Configured BukkaConfig instance.
            
        Examples:
            >>> args = argparse.Namespace(name="proj", dataset=None, backend="pyarrow")
            >>> config = BukkaConfig.from_args_and_config(args)
            >>> config.name
            'proj'
        """
        if config_dict is None:
            config_dict = DEFAULT_CONFIG
        
        # Helper to get value from args or config with proper precedence
        def get_value(arg_name: str, config_path: tuple[str, str], default: Any = None) -> Any:
            # First check CLI args (if not None)
            arg_value = getattr(args, arg_name, None)
            if arg_value is not None:
                return arg_value
            
            # Then check config dict
            section, key = config_path
            if config_dict and section in config_dict and key in config_dict[section]:
                config_value = config_dict[section][key]
                if config_value is not None:
                    return config_value
            
            # Finally use default
            return default
        
        return cls(
            name=get_value('name', ('project', 'name')),
            dataset=get_value('dataset', ('project', 'dataset')),
            target=get_value('target', ('project', 'target')),
            skip_venv=get_value('skip_venv', ('project', 'skip_venv'), False),
            enable_mlflow=get_value('enable_mlflow', ('project', 'enable_mlflow'), False),
            mlflow_tracking_uri=get_value('mlflow_tracking_uri', ('project', 'mlflow_tracking_uri')),
            backend=get_value('backend', ('data', 'backend'), 'pyarrow'),
            train_size=get_value('train_size', ('data', 'train_size'), 0.8),
            stratify=get_value('stratify', ('data', 'stratify'), True),
            strata=get_value('strata', ('data', 'strata')),
            problem_type=get_value('problem_type', ('problem', 'type'), 'auto'),
        )
    
    def validate(self) -> list[str]:
        """Validate all configuration fields.
        
        Returns:
            List of validation error messages (empty if valid).
            
        Examples:
            >>> config = BukkaConfig(name="my_project")
            >>> errors = config.validate()
            >>> len(errors)
            0
        """
        errors = []
        
        # Map fields to their validators
        validators = {
            'name': (ConfigValidator.validate_project_name, (ValueError,)),
            'dataset': (ConfigValidator.validate_dataset_path, (FileNotFoundError, ValueError)),
            'backend': (ConfigValidator.validate_backend, (ValueError,)),
            'problem_type': (ConfigValidator.validate_problem_type, (ValueError,)),
            'train_size': (ConfigValidator.validate_train_size, (ValueError,)),
        }
        
        for field_name, (validator, exceptions) in validators.items():
            value = getattr(self, field_name)
            if value is not None:  # Only validate non-None values
                try:
                    validator(value)
                except exceptions as e:
                    errors.append(str(e))
        
        return errors
    
    def to_project_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for Project constructor.
        
        Returns:
            Dictionary suitable for **kwargs unpacking into Project().
            
        Examples:
            >>> config = BukkaConfig(name="test")
            >>> kwargs = config.to_project_kwargs()
            >>> 'dataset_path' in kwargs
            True
        """
        return {
            'name': self.name,
            'dataset_path': self.dataset,
            'target_column': self.target,
            'skip_venv': self.skip_venv,
            'enable_mlflow': self.enable_mlflow,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'backend': self.backend,
            'problem_type': self.problem_type,
            'train_size': self.train_size,
            'stratify': self.stratify,
            'strata': self.strata,
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
            >>> ConfigValidator.validate_backend("pyarrow")
            'pyarrow'
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
        
        # Extract basename if it's a path (to allow absolute paths)
        from pathlib import Path
        basename = Path(name).name if ('/' in name or '\\' in name) else name
        
        # Check for potentially problematic characters in the basename only
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in basename for char in invalid_chars):
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
  enable_mlflow: {DEFAULT_CONFIG['project']['enable_mlflow']}  # Enable MLflow experiment tracking
  mlflow_tracking_uri: null  # MLflow tracking URI (default: mlruns/ in project)

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

    @staticmethod
    def merge_args_and_config(
        args: argparse.Namespace, 
        config_path: str | Path | None
    ) -> dict[str, Any]:
        """Merge CLI arguments with config file, CLI args take precedence.
        
        Args:
            args: Parsed command-line arguments.
            config_path: Path to YAML config file (optional).
            
        Returns:
            Merged configuration dictionary.
            
        Raises:
            FileNotFoundError: If config_path doesn't exist.
            ValueError: If config is invalid.
            
        Examples:
            >>> args = argparse.Namespace(name="my_proj", dataset=None)
            >>> config = ConfigManager.merge_args_and_config(args, None)
            >>> config['project']['name']
            'my_proj'
        """
        if config_path:
            # Load and validate config file
            config = ConfigManager.load_config(config_path)
        else:
            # Use defaults
            config = DEFAULT_CONFIG.copy()
        
        # Build argument mapping: CLI arg name -> (section, key) in config
        arg_mapping = {
            'name': ('project', 'name'),
            'dataset': ('project', 'dataset'),
            'target': ('project', 'target'),
            'skip_venv': ('project', 'skip_venv'),
            'backend': ('data', 'backend'),
            'train_size': ('data', 'train_size'),
            'stratify': ('data', 'stratify'),
            'strata': ('data', 'strata'),
            'problem_type': ('problem', 'type'),
        }
        
        # Override config with CLI arguments (if provided and not None)
        for arg_name, (section, key) in arg_mapping.items():
            cli_value = getattr(args, arg_name, None)
            if cli_value is not None:
                config[section][key] = cli_value
        
        return config

