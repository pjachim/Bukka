"""Unit tests for PyprojectTomlWriter class."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from bukka.coding.write_pyproject_toml import PyprojectTomlWriter, PYPROJECT_TOML_TEMPLATE


class TestPyprojectTomlTemplate:
    """Test suite for PYPROJECT_TOML_TEMPLATE constant."""

    def test_template_is_string(self):
        """Test that PYPROJECT_TOML_TEMPLATE is a non-empty string.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PYPROJECT_TOML_TEMPLATE
        >>> assert isinstance(PYPROJECT_TOML_TEMPLATE, str)
        >>> assert len(PYPROJECT_TOML_TEMPLATE) > 0
        """
        assert isinstance(PYPROJECT_TOML_TEMPLATE, str)
        assert len(PYPROJECT_TOML_TEMPLATE) > 0

    def test_template_has_placeholders(self):
        """Test that template contains expected placeholders.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PYPROJECT_TOML_TEMPLATE
        >>> assert '{project_name}' in PYPROJECT_TOML_TEMPLATE
        >>> assert '{requirements_file}' in PYPROJECT_TOML_TEMPLATE
        """
        assert '{project_name}' in PYPROJECT_TOML_TEMPLATE
        assert '{requirements_file}' in PYPROJECT_TOML_TEMPLATE

    def test_template_contains_toml_sections(self):
        """Test that template contains expected TOML sections.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PYPROJECT_TOML_TEMPLATE
        >>> assert '[project]' in PYPROJECT_TOML_TEMPLATE
        >>> assert '[build-system]' in PYPROJECT_TOML_TEMPLATE
        """
        assert '[project]' in PYPROJECT_TOML_TEMPLATE
        assert '[build-system]' in PYPROJECT_TOML_TEMPLATE
        assert '[tool.setuptools.dynamic]' in PYPROJECT_TOML_TEMPLATE
        assert '[tool.setuptools.packages.find]' in PYPROJECT_TOML_TEMPLATE

    def test_template_specifies_python_version(self):
        """Test that template specifies minimum Python version.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PYPROJECT_TOML_TEMPLATE
        >>> assert 'requires-python' in PYPROJECT_TOML_TEMPLATE
        >>> assert '3.10' in PYPROJECT_TOML_TEMPLATE
        """
        assert 'requires-python' in PYPROJECT_TOML_TEMPLATE
        assert '3.10' in PYPROJECT_TOML_TEMPLATE


class TestPyprojectTomlWriterInitialization:
    """Test suite for PyprojectTomlWriter initialization."""

    def test_writer_initialization_with_defaults(self):
        """Test PyprojectTomlWriter initialization with default project name.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> from pathlib import Path
        >>> file_manager = MagicMock()
        >>> file_manager.pyproject_toml_path = Path("pyproject.toml")
        >>> file_manager.requirements_path = Path("requirements.txt")
        >>> writer = PyprojectTomlWriter(file_manager)
        >>> assert writer.file_manager is file_manager
        """
        file_manager = MagicMock()
        file_manager.pyproject_toml_path = Path("pyproject.toml")
        file_manager.requirements_path = Path("requirements.txt")
        
        writer = PyprojectTomlWriter(file_manager)
        
        assert writer.file_manager is file_manager
        assert "project_name" in writer.kwargs
        assert writer.kwargs["project_name"] == "bukka_project"
        assert writer.kwargs["requirements_file"] == "requirements.txt"

    def test_writer_initialization_with_custom_project_name(self):
        """Test PyprojectTomlWriter initialization with custom project name.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> from pathlib import Path
        >>> file_manager = MagicMock()
        >>> file_manager.pyproject_toml_path = Path("pyproject.toml")
        >>> file_manager.requirements_path = Path("requirements.txt")
        >>> writer = PyprojectTomlWriter(file_manager, project_name="my_project")
        >>> assert writer.kwargs["project_name"] == "my_project"
        """
        file_manager = MagicMock()
        file_manager.pyproject_toml_path = Path("pyproject.toml")
        file_manager.requirements_path = Path("requirements.txt")
        
        writer = PyprojectTomlWriter(
            file_manager,
            project_name="my_custom_project"
        )
        
        assert writer.kwargs["project_name"] == "my_custom_project"

    def test_writer_inherits_from_template_base_class(self):
        """Test that PyprojectTomlWriter inherits from TemplateBaseClass.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from bukka.coding.utils.template_handler import TemplateBaseClass
        >>> from unittest.mock import MagicMock
        >>> from pathlib import Path
        >>> file_manager = MagicMock()
        >>> file_manager.pyproject_toml_path = Path("pyproject.toml")
        >>> file_manager.requirements_path = Path("requirements.txt")
        >>> writer = PyprojectTomlWriter(file_manager)
        >>> assert isinstance(writer, TemplateBaseClass)
        """
        from bukka.coding.utils.template_handler import TemplateBaseClass
        
        file_manager = MagicMock()
        file_manager.pyproject_toml_path = Path("pyproject.toml")
        file_manager.requirements_path = Path("requirements.txt")
        
        writer = PyprojectTomlWriter(file_manager)
        
        assert isinstance(writer, TemplateBaseClass)

    def test_writer_sets_correct_output_path(self):
        """Test that writer sets output_path from file_manager.
        
        Examples
        --------
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> from pathlib import Path
        >>> file_manager = MagicMock()
        >>> file_manager.pyproject_toml_path = Path("/project/pyproject.toml")
        >>> file_manager.requirements_path = Path("requirements.txt")
        >>> writer = PyprojectTomlWriter(file_manager)
        >>> assert writer.output_path == Path("/project/pyproject.toml")
        """
        file_manager = MagicMock()
        file_manager.pyproject_toml_path = Path("/project/pyproject.toml")
        file_manager.requirements_path = Path("requirements.txt")
        
        writer = PyprojectTomlWriter(file_manager)
        
        assert writer.output_path == Path("/project/pyproject.toml")


class TestPyprojectTomlWriterWriteCode:
    """Test suite for PyprojectTomlWriter.write_code method."""

    def test_write_code_creates_file(self):
        """Test that write_code creates a pyproject.toml file.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "pyproject.toml"
        ...     file_manager = MagicMock()
        ...     file_manager.pyproject_toml_path = output_path
        ...     file_manager.requirements_path = Path("requirements.txt")
        ...     writer = PyprojectTomlWriter(file_manager, project_name="test_proj")
        ...     writer.write_code()
        ...     assert output_path.exists()
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pyproject.toml"
            file_manager = MagicMock()
            file_manager.pyproject_toml_path = output_path
            file_manager.requirements_path = Path("requirements.txt")
            
            writer = PyprojectTomlWriter(
                file_manager,
                project_name="test_project"
            )
            writer.write_code()
            
            assert output_path.exists()

    def test_write_code_contains_project_name(self):
        """Test that generated file contains the project name.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "pyproject.toml"
        ...     file_manager = MagicMock()
        ...     file_manager.pyproject_toml_path = output_path
        ...     file_manager.requirements_path = Path("requirements.txt")
        ...     writer = PyprojectTomlWriter(file_manager, project_name="my_ml_project")
        ...     writer.write_code()
        ...     content = output_path.read_text()
        ...     assert 'name = "my_ml_project"' in content
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pyproject.toml"
            file_manager = MagicMock()
            file_manager.pyproject_toml_path = output_path
            file_manager.requirements_path = Path("requirements.txt")
            
            writer = PyprojectTomlWriter(
                file_manager,
                project_name="my_ml_project"
            )
            writer.write_code()
            
            content = output_path.read_text()
            assert 'name = "my_ml_project"' in content

    def test_write_code_contains_requirements_file(self):
        """Test that generated file references the requirements file.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "pyproject.toml"
        ...     file_manager = MagicMock()
        ...     file_manager.pyproject_toml_path = output_path
        ...     file_manager.requirements_path = Path("requirements.txt")
        ...     writer = PyprojectTomlWriter(file_manager)
        ...     writer.write_code()
        ...     content = output_path.read_text()
        ...     assert 'requirements.txt' in content
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pyproject.toml"
            file_manager = MagicMock()
            file_manager.pyproject_toml_path = output_path
            file_manager.requirements_path = Path("custom_requirements.txt")
            
            writer = PyprojectTomlWriter(file_manager)
            writer.write_code()
            
            content = output_path.read_text()
            assert 'custom_requirements.txt' in content

    def test_write_code_contains_build_system(self):
        """Test that generated file contains build system configuration.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "pyproject.toml"
        ...     file_manager = MagicMock()
        ...     file_manager.pyproject_toml_path = output_path
        ...     file_manager.requirements_path = Path("requirements.txt")
        ...     writer = PyprojectTomlWriter(file_manager)
        ...     writer.write_code()
        ...     content = output_path.read_text()
        ...     assert '[build-system]' in content
        ...     assert 'setuptools' in content
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pyproject.toml"
            file_manager = MagicMock()
            file_manager.pyproject_toml_path = output_path
            file_manager.requirements_path = Path("requirements.txt")
            
            writer = PyprojectTomlWriter(file_manager)
            writer.write_code()
            
            content = output_path.read_text()
            assert '[build-system]' in content
            assert 'setuptools' in content

    def test_write_code_is_valid_toml(self):
        """Test that generated file is valid TOML format.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_pyproject_toml import PyprojectTomlWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "pyproject.toml"
        ...     file_manager = MagicMock()
        ...     file_manager.pyproject_toml_path = output_path
        ...     file_manager.requirements_path = Path("requirements.txt")
        ...     writer = PyprojectTomlWriter(file_manager, project_name="test")
        ...     writer.write_code()
        ...     # Try to parse the TOML file
        ...     import tomllib
        ...     content = output_path.read_text()
        ...     tomllib.loads(content)
        """
        import tomllib
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pyproject.toml"
            file_manager = MagicMock()
            file_manager.pyproject_toml_path = output_path
            file_manager.requirements_path = Path("requirements.txt")
            
            writer = PyprojectTomlWriter(
                file_manager,
                project_name="test_project"
            )
            writer.write_code()
            
            # Verify the file can be parsed as TOML
            content = output_path.read_text()
            parsed = tomllib.loads(content)
            
            # Verify structure
            assert 'project' in parsed
            assert 'build-system' in parsed
            assert parsed['project']['name'] == 'test_project'
