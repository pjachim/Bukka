"""Unit tests for EnvironmentBuilder class."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from bukka.environment.environment import EnvironmentBuilder


class TestEnvironmentBuilderInitialization:
    """Test suite for EnvironmentBuilder initialization."""

    def test_initialization_with_file_manager(self):
        """Test EnvironmentBuilder initialization with FileManager.
        
        Examples
        --------
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> from unittest.mock import MagicMock
        >>> file_manager = MagicMock()
        >>> env_builder = EnvironmentBuilder(file_manager)
        >>> assert env_builder.file_manager is file_manager
        """
        file_manager = MagicMock()
        env_builder = EnvironmentBuilder(file_manager)
        
        assert env_builder.file_manager is file_manager


class TestEnvironmentBuilderBuildEnvironment:
    """Test suite for EnvironmentBuilder.build_environment method."""

    @patch.object(EnvironmentBuilder, '_install_package_editable')
    @patch.object(EnvironmentBuilder, '_install_packages')
    @patch.object(EnvironmentBuilder, '_build_venv')
    def test_build_environment_calls_all_steps(
        self, mock_build_venv, mock_install_packages, mock_install_editable
    ):
        """Test that build_environment calls all setup steps in order.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> file_manager = MagicMock()
        >>> with patch.object(EnvironmentBuilder, '_build_venv'):
        ...     with patch.object(EnvironmentBuilder, '_install_packages'):
        ...         with patch.object(EnvironmentBuilder, '_install_package_editable'):
        ...             env_builder = EnvironmentBuilder(file_manager)
        ...             env_builder.build_environment()
        """
        file_manager = MagicMock()
        env_builder = EnvironmentBuilder(file_manager)
        
        env_builder.build_environment()
        
        # Verify all steps were called
        mock_build_venv.assert_called_once()
        mock_install_packages.assert_called_once()
        mock_install_editable.assert_called_once()

    @patch.object(EnvironmentBuilder, '_install_package_editable')
    @patch.object(EnvironmentBuilder, '_install_packages')
    @patch.object(EnvironmentBuilder, '_build_venv')
    def test_build_environment_order_of_operations(
        self, mock_build_venv, mock_install_packages, mock_install_editable
    ):
        """Test that build_environment executes steps in correct order.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock, call
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> # Verify venv is built before packages are installed
        """
        file_manager = MagicMock()
        env_builder = EnvironmentBuilder(file_manager)
        
        # Use a manager to track call order
        manager = MagicMock()
        manager.attach_mock(mock_build_venv, 'build_venv')
        manager.attach_mock(mock_install_packages, 'install_packages')
        manager.attach_mock(mock_install_editable, 'install_editable')
        
        env_builder.build_environment()
        
        # Verify order: build_venv, then install_packages, then install_editable
        expected_calls = [
            call.build_venv(),
            call.install_packages(),
            call.install_editable()
        ]
        assert manager.mock_calls == expected_calls


class TestEnvironmentBuilderBuildVenv:
    """Test suite for EnvironmentBuilder._build_venv method."""

    @patch('bukka.environment.environment.venv.EnvBuilder')
    def test_build_venv_creates_env_builder(self, mock_env_builder_class):
        """Test that _build_venv creates a venv.EnvBuilder with pip.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> with patch('bukka.environment.environment.venv.EnvBuilder') as mock:
        ...     file_manager = MagicMock()
        ...     file_manager.virtual_env = Path("/project/.venv")
        ...     env_builder = EnvironmentBuilder(file_manager)
        ...     env_builder._build_venv()
        ...     mock.assert_called_once_with(with_pip=True)
        """
        mock_venv_instance = MagicMock()
        mock_env_builder_class.return_value = mock_venv_instance
        
        file_manager = MagicMock()
        file_manager.virtual_env = Path("/project/.venv")
        
        env_builder = EnvironmentBuilder(file_manager)
        env_builder._build_venv()
        
        # Verify EnvBuilder was created with with_pip=True
        mock_env_builder_class.assert_called_once_with(with_pip=True)
        
        # Verify create was called with the virtual_env path
        mock_venv_instance.create.assert_called_once_with(
            env_dir=file_manager.virtual_env
        )


class TestEnvironmentBuilderInstallPackages:
    """Test suite for EnvironmentBuilder._install_packages method."""

    @patch('bukka.environment.environment.subprocess.run')
    def test_install_packages_writes_requirements(self, mock_subprocess_run):
        """Test that _install_packages writes requirements file.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from unittest.mock import patch, MagicMock
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     requirements_path = Path(tmp_dir) / "requirements.txt"
        ...     file_manager = MagicMock()
        ...     file_manager.requirements_path = requirements_path
        ...     file_manager.python_path = Path("python")
        ...     with patch('bukka.environment.environment.subprocess.run'):
        ...         env_builder = EnvironmentBuilder(file_manager)
        ...         env_builder._install_packages()
        ...         assert requirements_path.exists()
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            requirements_path = Path(tmp_dir) / "requirements.txt"
            python_path = Path(tmp_dir) / "python"
            
            file_manager = MagicMock()
            file_manager.requirements_path = requirements_path
            file_manager.python_path = python_path
            
            env_builder = EnvironmentBuilder(file_manager)
            env_builder._install_packages()
            
            # Verify requirements file was created
            assert requirements_path.exists()
            
            # Verify it contains expected packages
            content = requirements_path.read_text()
            assert 'scikit-learn' in content
            assert 'narwhals' in content
            assert 'pyarrow' in content

    @patch('bukka.environment.environment.subprocess.run')
    def test_install_packages_calls_pip(self, mock_subprocess_run):
        """Test that _install_packages calls pip install.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> from pathlib import Path
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> with patch('bukka.environment.environment.subprocess.run') as mock_run:
        ...     file_manager = MagicMock()
        ...     file_manager.requirements_path = Path("requirements.txt")
        ...     file_manager.python_path = Path("/venv/bin/python")
        ...     env_builder = EnvironmentBuilder(file_manager)
        ...     # Mock file creation
        ...     with patch('builtins.open'):
        ...         env_builder._install_packages()
        ...     # Verify pip was called
        ...     mock_run.assert_called_once()
        """
        file_manager = MagicMock()
        file_manager.requirements_path = Path("requirements.txt")
        file_manager.python_path = Path("/venv/bin/python")
        
        env_builder = EnvironmentBuilder(file_manager)
        env_builder._install_packages()
        
        # Verify subprocess.run was called
        mock_subprocess_run.assert_called_once()
        
        # Verify the command includes pip install -r requirements.txt
        call_args = mock_subprocess_run.call_args[0][0]
        assert str(file_manager.python_path) in call_args
        assert '-m' in call_args
        assert 'pip' in call_args
        assert 'install' in call_args
        assert '-r' in call_args
        assert str(file_manager.requirements_path) in call_args


class TestEnvironmentBuilderInstallPackageEditable:
    """Test suite for EnvironmentBuilder._install_package_editable method."""

    @patch('bukka.environment.environment.subprocess.run')
    def test_install_package_editable_calls_pip(self, mock_subprocess_run):
        """Test that _install_package_editable calls pip install -e.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> from pathlib import Path
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> with patch('bukka.environment.environment.subprocess.run') as mock_run:
        ...     file_manager = MagicMock()
        ...     file_manager.python_path = Path("/venv/bin/python")
        ...     file_manager.project_path = Path("/project")
        ...     env_builder = EnvironmentBuilder(file_manager)
        ...     env_builder._install_package_editable()
        ...     mock_run.assert_called_once()
        """
        file_manager = MagicMock()
        file_manager.python_path = Path("/venv/bin/python")
        file_manager.project_path = Path("/project")
        
        env_builder = EnvironmentBuilder(file_manager)
        env_builder._install_package_editable()
        
        # Verify subprocess.run was called
        mock_subprocess_run.assert_called_once()
        
        # Verify the command includes pip install -e <project_path>
        call_args = mock_subprocess_run.call_args[0][0]
        assert str(file_manager.python_path) in call_args
        assert '-m' in call_args
        assert 'pip' in call_args
        assert 'install' in call_args
        assert '-e' in call_args
        assert str(file_manager.project_path) in call_args

    @patch('bukka.environment.environment.subprocess.run')
    def test_install_package_editable_uses_correct_python(self, mock_subprocess_run):
        """Test that editable install uses the venv's Python.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> from pathlib import Path
        >>> from bukka.environment.environment import EnvironmentBuilder
        >>> with patch('bukka.environment.environment.subprocess.run') as mock_run:
        ...     file_manager = MagicMock()
        ...     python_path = Path("/custom/venv/bin/python3.10")
        ...     file_manager.python_path = python_path
        ...     file_manager.project_path = Path("/project")
        ...     env_builder = EnvironmentBuilder(file_manager)
        ...     env_builder._install_package_editable()
        ...     # First argument should be the custom python path
        ...     call_args = mock_run.call_args[0][0]
        ...     assert str(python_path) == call_args[0]
        """
        file_manager = MagicMock()
        python_path = Path("/custom/venv/bin/python3.10")
        file_manager.python_path = python_path
        file_manager.project_path = Path("/project")
        
        env_builder = EnvironmentBuilder(file_manager)
        env_builder._install_package_editable()
        
        # Verify the custom Python path is used
        call_args = mock_subprocess_run.call_args[0][0]
        assert str(python_path) == call_args[0]
