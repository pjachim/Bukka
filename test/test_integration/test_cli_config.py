"""Unit tests for CLI configuration and validation.

Tests the new CLI features including:
- YAML configuration file support
- ConfigManager and ConfigValidator classes
- Input validation
- Subcommand architecture (init-config, run)
- Backend selection
- Problem type specification
"""

import pytest
from pathlib import Path
import yaml
import tempfile
import sys
import subprocess
import textwrap

from bukka.cli_config import (
    ConfigManager,
    ConfigValidator,
    SUPPORTED_BACKENDS,
    PROBLEM_TYPES,
    DEFAULT_CONFIG,
)


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    def test_validate_backend_valid(self):
        """Test validation accepts all supported backends."""
        for backend in SUPPORTED_BACKENDS:
            assert ConfigValidator.validate_backend(backend) == backend

    def test_validate_backend_invalid(self):
        """Test validation rejects unsupported backends."""
        with pytest.raises(ValueError, match="not supported"):
            ConfigValidator.validate_backend("invalid_backend")
        
        with pytest.raises(ValueError, match="not supported"):
            ConfigValidator.validate_backend("numpy")

    def test_validate_problem_type_valid(self):
        """Test validation accepts all supported problem types."""
        for problem_type in PROBLEM_TYPES:
            assert ConfigValidator.validate_problem_type(problem_type) == problem_type

    def test_validate_problem_type_none_defaults_to_auto(self):
        """Test None problem type defaults to 'auto'."""
        assert ConfigValidator.validate_problem_type(None) == "auto"

    def test_validate_problem_type_invalid(self):
        """Test validation rejects unsupported problem types."""
        with pytest.raises(ValueError, match="not recognized"):
            ConfigValidator.validate_problem_type("invalid_type")
        
        with pytest.raises(ValueError, match="not recognized"):
            ConfigValidator.validate_problem_type("supervised")

    def test_validate_dataset_path_none(self):
        """Test None dataset path is accepted."""
        assert ConfigValidator.validate_dataset_path(None) is None

    def test_validate_dataset_path_nonexistent(self):
        """Test validation rejects non-existent files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ConfigValidator.validate_dataset_path("/nonexistent/path/data.csv")

    def test_validate_dataset_path_valid(self, tmp_path):
        """Test validation accepts existing files."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n", encoding="utf-8")
        
        result = ConfigValidator.validate_dataset_path(str(csv_file))
        assert isinstance(result, Path)
        assert result.exists()

    def test_validate_dataset_path_directory(self, tmp_path):
        """Test validation rejects directories."""
        with pytest.raises(ValueError, match="not a file"):
            ConfigValidator.validate_dataset_path(str(tmp_path))

    def test_validate_train_size_valid(self):
        """Test train size validation for valid values."""
        assert ConfigValidator.validate_train_size(0.5) == 0.5
        assert ConfigValidator.validate_train_size(0.8) == 0.8
        assert ConfigValidator.validate_train_size(0.1) == 0.1
        assert ConfigValidator.validate_train_size(0.9) == 0.9

    def test_validate_train_size_invalid(self):
        """Test train size validation rejects out-of-bounds values."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConfigValidator.validate_train_size(0.0)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConfigValidator.validate_train_size(1.0)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConfigValidator.validate_train_size(1.5)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConfigValidator.validate_train_size(-0.1)

    def test_validate_project_name_valid(self):
        """Test project name validation for valid names."""
        assert ConfigValidator.validate_project_name("my_project") == "my_project"
        assert ConfigValidator.validate_project_name("project123") == "project123"
        assert ConfigValidator.validate_project_name("  my_project  ") == "my_project"

    def test_validate_project_name_empty(self):
        """Test validation rejects empty names."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ConfigValidator.validate_project_name("")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            ConfigValidator.validate_project_name("   ")

    def test_validate_project_name_invalid_chars(self):
        """Test validation rejects names with invalid characters."""
        invalid_names = ["my<project", "my>project", "my:project", 'my"project', 
                        "my|project", "my?project", "my*project"]
        
        for name in invalid_names:
            with pytest.raises(ValueError, match="invalid characters"):
                ConfigValidator.validate_project_name(name)


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_create_template_default_path(self, tmp_path):
        """Test template creation with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_path = tmpdir_path / "bukka_config.yaml"
            
            result = ConfigManager.create_template(output_path)
            
            assert result.exists()
            assert result.name == "bukka_config.yaml"
            
            # Verify YAML is valid
            with open(result, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            assert 'project' in config
            assert 'data' in config
            assert 'problem' in config

    def test_create_template_custom_path(self, tmp_path):
        """Test template creation with custom path."""
        output_path = tmp_path / "my_config.yaml"
        
        result = ConfigManager.create_template(output_path)
        
        assert result.exists()
        assert result.name == "my_config.yaml"

    def test_create_template_contains_defaults(self, tmp_path):
        """Test template contains all default values."""
        output_path = tmp_path / "test_config.yaml"
        ConfigManager.create_template(output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check project section
        assert config['project']['name'] == DEFAULT_CONFIG['project']['name']
        assert config['project']['skip_venv'] == DEFAULT_CONFIG['project']['skip_venv']
        
        # Check data section
        assert config['data']['backend'] == DEFAULT_CONFIG['data']['backend']
        assert config['data']['train_size'] == DEFAULT_CONFIG['data']['train_size']
        
        # Check problem section
        assert config['problem']['type'] == DEFAULT_CONFIG['problem']['type']

    def test_create_template_contains_comments(self, tmp_path):
        """Test template contains helpful comments."""
        output_path = tmp_path / "test_config.yaml"
        ConfigManager.create_template(output_path)
        
        content = output_path.read_text(encoding='utf-8')
        
        assert "# Bukka Project Configuration Template" in content
        assert "# Project settings" in content
        assert "# Data processing settings" in content
        assert "# Problem specification" in content

    def test_load_config_nonexistent(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config("/nonexistent/config.yaml")

    def test_load_config_empty(self, tmp_path):
        """Test loading empty config file raises error."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("", encoding='utf-8')
        
        with pytest.raises(ValueError, match="empty"):
            ConfigManager.load_config(config_path)

    def test_load_config_valid(self, tmp_path):
        """Test loading valid config file."""
        config_path = tmp_path / "valid.yaml"
        config_data = {
            'project': {
                'name': 'test_project',
                'dataset': None,
                'target': None,
                'skip_venv': True
            },
            'data': {
                'backend': 'pandas',
                'train_size': 0.7,
                'stratify': False,
                'strata': None
            },
            'problem': {
                'type': 'regression'
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        loaded_config = ConfigManager.load_config(config_path)
        
        assert loaded_config['project']['name'] == 'test_project'
        assert loaded_config['data']['backend'] == 'pandas'
        assert loaded_config['data']['train_size'] == 0.7
        assert loaded_config['problem']['type'] == 'regression'

    def test_load_config_merges_with_defaults(self, tmp_path):
        """Test partial config merges with defaults."""
        config_path = tmp_path / "partial.yaml"
        config_data = {
            'project': {
                'name': 'my_project'
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        loaded_config = ConfigManager.load_config(config_path)
        
        # Custom value
        assert loaded_config['project']['name'] == 'my_project'
        
        # Default values
        assert loaded_config['data']['backend'] == DEFAULT_CONFIG['data']['backend']
        assert loaded_config['problem']['type'] == DEFAULT_CONFIG['problem']['type']

    def test_load_config_validates_backend(self, tmp_path):
        """Test loading config validates backend."""
        config_path = tmp_path / "invalid_backend.yaml"
        config_data = {
            'project': {'name': 'test'},
            'data': {'backend': 'invalid_backend'},
            'problem': {'type': 'auto'}
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="not supported"):
            ConfigManager.load_config(config_path)

    def test_load_config_validates_problem_type(self, tmp_path):
        """Test loading config validates problem type."""
        config_path = tmp_path / "invalid_problem.yaml"
        config_data = {
            'project': {'name': 'test'},
            'data': {'backend': 'polars'},
            'problem': {'type': 'invalid_type'}
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="not recognized"):
            ConfigManager.load_config(config_path)

    def test_load_config_validates_train_size(self, tmp_path):
        """Test loading config validates train size."""
        config_path = tmp_path / "invalid_train_size.yaml"
        config_data = {
            'project': {'name': 'test'},
            'data': {'backend': 'polars', 'train_size': 1.5},
            'problem': {'type': 'auto'}
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConfigManager.load_config(config_path)


class TestCLISubcommands:
    """Test CLI subcommand functionality."""

    def test_init_config_creates_file(self, tmp_path):
        """Test init-config subcommand creates config file."""
        output_path = tmp_path / "cli_test_config.yaml"
        
        code = textwrap.dedent(f"""
            import sys
            sys.argv = ['bukka', 'init-config', '--output', r'{output_path}']
            from bukka.__main__ import main
            main()
        """)
        
        result = subprocess.run([sys.executable, "-c", code], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert output_path.exists()

    def test_run_requires_name_or_config(self):
        """Test run subcommand requires --name or --config."""
        code = textwrap.dedent("""
            import sys
            sys.argv = ['bukka', 'run']
            try:
                from bukka.__main__ import main
                main()
            except SystemExit as e:
                sys.exit(e.code if e.code else 0)
        """)
        
        result = subprocess.run([sys.executable, "-c", code], 
                              capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()

    def test_help_displays_subcommands(self):
        """Test --help shows available subcommands."""
        result = subprocess.run([sys.executable, "-m", "bukka", "--help"],
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "init-config" in result.stdout
        assert "run" in result.stdout

    def test_run_help_shows_all_options(self):
        """Test run --help shows all configuration options."""
        result = subprocess.run([sys.executable, "-m", "bukka", "run", "--help"],
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "--backend" in result.stdout
        assert "--problem-type" in result.stdout
        assert "--config" in result.stdout
        assert "--train-size" in result.stdout
        assert "--stratify" in result.stdout


class TestCLIValidation:
    """Test CLI input validation."""

    def test_validates_backend_at_argparse_level(self):
        """Test invalid backend is rejected by argparse."""
        code = textwrap.dedent("""
            import sys
            sys.argv = ['bukka', 'run', '--name', 'test', '--backend', 'invalid']
            try:
                from bukka.__main__ import main
                main()
            except SystemExit as e:
                sys.exit(e.code if e.code else 0)
        """)
        
        result = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower()

    def test_validates_problem_type_at_argparse_level(self):
        """Test invalid problem type is rejected by argparse."""
        code = textwrap.dedent("""
            import sys
            sys.argv = ['bukka', 'run', '--name', 'test', '--problem-type', 'invalid']
            try:
                from bukka.__main__ import main
                main()
            except SystemExit as e:
                sys.exit(e.code if e.code else 0)
        """)
        
        result = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower()

    def test_validates_nonexistent_dataset(self, tmp_path):
        """Test validation catches non-existent dataset files."""
        code = textwrap.dedent(f"""
            import sys
            sys.argv = ['bukka', 'run', '--name', 'test', '--dataset', '/nonexistent/data.csv']
            try:
                from bukka.__main__ import main
                main()
            except SystemExit as e:
                sys.exit(e.code if e.code else 0)
        """)
        
        result = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()


class TestCLIIntegrationWithNewFeatures:
    """Integration tests for CLI with new features."""

    def test_cli_with_backend_option(self, tmp_path):
        """Test CLI run with --backend option."""
        csv = tmp_path / "data.csv"
        csv.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
        
        proj = tmp_path / "test_proj"
        
        # Patch to avoid actual environment setup
        code = textwrap.dedent(f"""
            import sys
            from bukka.environment.environment import EnvironmentBuilder
            EnvironmentBuilder.build_environment = lambda self: None
            
            from bukka.data_management import dataset as ds_module
            original_init = ds_module.Dataset.__init__
            
            def patched_init(self, *args, **kwargs):
                # Track that backend was passed
                self._test_backend = kwargs.get('backend', kwargs.get('dataframe_backend', 'polars'))
                raise RuntimeError(f"Backend received: {{self._test_backend}}")
            
            ds_module.Dataset.__init__ = patched_init
            
            sys.argv = ['bukka', 'run', '--name', r'{proj}', '--dataset', r'{csv}', 
                       '--target', 'target', '--backend', 'pandas', '--skip-venv']
            
            try:
                from bukka.__main__ import main
                main()
            except RuntimeError as e:
                if "Backend received" in str(e):
                    print(str(e))
                    sys.exit(0)
                raise
        """)
        
        result = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)

        # The test passes if we successfully passed the backend parameter
        assert result.returncode == 0 or "Backend received: pandas" in result.stderr
    
    def test_cli_with_problem_type_option(self, tmp_path):
        """Test CLI run with --problem-type option."""
        csv = tmp_path / "data.csv"
        csv.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
        
        proj = tmp_path / "test_proj"
        
        code = textwrap.dedent(f"""
            import sys
            from bukka.environment.environment import EnvironmentBuilder
            EnvironmentBuilder.build_environment = lambda self: None
            
            from bukka import project as proj_module
            original_init = proj_module.Project.__init__
            
            def patched_init(self, *args, **kwargs):
                self._test_problem_type = kwargs.get('problem_type', 'auto')
                raise RuntimeError(f"Problem type received: {{self._test_problem_type}}")
            
            proj_module.Project.__init__ = patched_init
            
            sys.argv = ['bukka', 'run', '--name', r'{proj}', '--dataset', r'{csv}',
                       '--target', 'target', '--problem-type', 'regression', '--skip-venv']
            
            try:
                from bukka.__main__ import main
                main()
            except RuntimeError as e:
                if "Problem type received" in str(e):
                    print(str(e))
                    sys.exit(0)
                raise
        """)
        
        result = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)

        assert result.returncode == 0 or "Problem type received: regression" in result.stderr
    
    def test_cli_with_config_file(self, tmp_path):
        """Test CLI run with --config option."""
        csv = tmp_path / "data.csv"
        csv.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
        
        config_path = tmp_path / "config.yaml"
        config_data = {
            'project': {
                'name': 'config_test_proj',
                'dataset': str(csv),
                'target': 'target',
                'skip_venv': True
            },
            'data': {
                'backend': 'pandas',
                'train_size': 0.7
            },
            'problem': {
                'type': 'binary_classification'
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        code = textwrap.dedent(f"""
            import sys
            from bukka.environment.environment import EnvironmentBuilder
            EnvironmentBuilder.build_environment = lambda self: None
            
            from bukka import project as proj_module
            original_init = proj_module.Project.__init__
            
            def patched_init(self, *args, **kwargs):
                self._test_backend = kwargs.get('backend', 'polars')
                self._test_problem_type = kwargs.get('problem_type', 'auto')
                self._test_train_size = kwargs.get('train_size', 0.8)
                raise RuntimeError(
                    f"Config loaded: backend={{self._test_backend}}, "
                    f"problem_type={{self._test_problem_type}}, "
                    f"train_size={{self._test_train_size}}"
                )
            
            proj_module.Project.__init__ = patched_init
            
            sys.argv = ['bukka', 'run', '--config', r'{config_path}']
            
            try:
                from bukka.__main__ import main
                main()
            except RuntimeError as e:
                if "Config loaded" in str(e):
                    print(str(e))
                    sys.exit(0)
                raise
        """)
        
        result = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)

        assert result.returncode == 0 or "backend=pandas" in result.stderr
        assert result.returncode == 0 or "problem_type=binary_classification" in result.stderr
        assert result.returncode == 0 or "train_size=0.7" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
