"""Unit tests for ConfigWriter class."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from bukka.coding.write_config import ConfigWriter, CONFIG_TEMPLATE


class TestConfigTemplate:
    """Test suite for CONFIG_TEMPLATE constant."""

    def test_config_template_is_string(self):
        """Test that CONFIG_TEMPLATE is a non-empty string.
        
        Examples
        --------
        >>> from bukka.coding.write_config import CONFIG_TEMPLATE
        >>> assert isinstance(CONFIG_TEMPLATE, str)
        >>> assert len(CONFIG_TEMPLATE) > 0
        """
        assert isinstance(CONFIG_TEMPLATE, str)
        assert len(CONFIG_TEMPLATE) > 0

    def test_config_template_has_placeholders(self):
        """Test that CONFIG_TEMPLATE contains expected placeholders.
        
        Examples
        --------
        >>> from bukka.coding.write_config import CONFIG_TEMPLATE
        >>> assert '{backend_name}' in CONFIG_TEMPLATE
        >>> assert '{train_relative_path}' in CONFIG_TEMPLATE
        """
        assert '{backend_name}' in CONFIG_TEMPLATE
        assert '{train_relative_path}' in CONFIG_TEMPLATE
        assert '{test_relative_path}' in CONFIG_TEMPLATE

    def test_config_template_contains_expected_constants(self):
        """Test that CONFIG_TEMPLATE defines expected configuration constants.
        
        Examples
        --------
        >>> from bukka.coding.write_config import CONFIG_TEMPLATE
        >>> assert 'DATAFRAME_BACKEND' in CONFIG_TEMPLATE
        >>> assert 'TRAIN_DATASET_PATH' in CONFIG_TEMPLATE
        """
        assert 'DATAFRAME_BACKEND' in CONFIG_TEMPLATE
        assert 'CURRENT_DIR' in CONFIG_TEMPLATE
        assert 'TRAIN_DATASET_PATH' in CONFIG_TEMPLATE
        assert 'TEST_DATASET_PATH' in CONFIG_TEMPLATE


class TestConfigWriterInitialization:
    """Test suite for ConfigWriter initialization."""

    def test_config_writer_initialization(self):
        """Test ConfigWriter initialization with required arguments.
        
        Examples
        --------
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> file_manager = MagicMock()
        >>> writer = ConfigWriter(
        ...     output_path="config.py",
        ...     backend_name="pyarrow",
        ...     file_manager=file_manager
        ... )
        >>> assert writer.output_path == "config.py"
        >>> assert writer.backend_name == "polars"
        """
        file_manager = MagicMock()
        writer = ConfigWriter(
            output_path="config.py",
            backend_name="pyarrow",
            file_manager=file_manager
        )
        
        assert writer.output_path == "config.py"
        assert writer.backend_name == "polars"
        assert writer.file_manager is file_manager

    def test_config_writer_accepts_different_backends(self):
        """Test ConfigWriter initialization with different backend names.
        
        Examples
        --------
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> file_manager = MagicMock()
        >>> writer_polars = ConfigWriter("config.py", "polars", file_manager)
        >>> writer_pandas = ConfigWriter("config.py", "pandas", file_manager)
        >>> assert writer_polars.backend_name == "polars"
        >>> assert writer_pandas.backend_name == "pandas"
        """
        file_manager = MagicMock()
        
        writer_pyarrow = ConfigWriter("config.py", "pyarrow", file_manager)
        assert writer_pyarrow.backend_name == "pyarrow"
        
        writer_modin = ConfigWriter("config.py", "modin", file_manager)
        assert writer_modin.backend_name == "modin"


class TestConfigWriterWriteConfig:
    """Test suite for ConfigWriter.write_config method."""

    def test_write_config_creates_file(self):
        """Test that write_config creates a configuration file.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "config.py"
        ...     file_manager = MagicMock()
        ...     file_manager.train_file_relative_path = Path("data/train.parquet")
        ...     file_manager.test_file_relative_path = Path("data/test.parquet")
        ...     writer = ConfigWriter(str(output_path), "pyarrow", file_manager)
        ...     writer.write_config()
        ...     assert output_path.exists()
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.py"
            file_manager = MagicMock()
            file_manager.train_file_relative_path = Path("data/train.parquet")
            file_manager.test_file_relative_path = Path("data/test.parquet")
            
            writer = ConfigWriter(
                str(output_path),
                "pyarrow",
                file_manager
            )
            writer.write_config()
            
            assert output_path.exists()

    def test_write_config_contains_backend_name(self):
        """Test that generated config file contains the backend name.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "config.py"
        ...     file_manager = MagicMock()
        ...     file_manager.train_file_relative_path = Path("data/train.parquet")
        ...     file_manager.test_file_relative_path = Path("data/test.parquet")
        ...     writer = ConfigWriter(str(output_path), "pyarrow", file_manager)
        ...     writer.write_config()
        ...     content = output_path.read_text()
        ...     assert "DATAFRAME_BACKEND = 'pandas'" in content
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.py"
            file_manager = MagicMock()
            file_manager.train_file_relative_path = Path("data/train.parquet")
            file_manager.test_file_relative_path = Path("data/test.parquet")
            
            writer = ConfigWriter(
                str(output_path),
                "pyarrow",
                file_manager
            )
            writer.write_config()
            
            content = output_path.read_text()
            assert "DATAFRAME_BACKEND = 'pyarrow'" in content

    def test_write_config_contains_dataset_paths(self):
        """Test that generated config file contains dataset paths.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "config.py"
        ...     file_manager = MagicMock()
        ...     file_manager.train_file_relative_path = Path("data/train/train.parquet")
        ...     file_manager.test_file_relative_path = Path("data/test/test.parquet")
        ...     writer = ConfigWriter(str(output_path), "pyarrow", file_manager)
        ...     writer.write_config()
        ...     content = output_path.read_text()
        ...     assert "TRAIN_DATASET_PATH" in content
        ...     assert "TEST_DATASET_PATH" in content
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.py"
            file_manager = MagicMock()
            file_manager.train_file_relative_path = Path("data/train/train.parquet")
            file_manager.test_file_relative_path = Path("data/test/test.parquet")
            
            writer = ConfigWriter(
                str(output_path),
                "pyarrow",
                file_manager
            )
            writer.write_config()
            
            content = output_path.read_text()
            assert "TRAIN_DATASET_PATH" in content
            assert "TEST_DATASET_PATH" in content
            # Verify paths use forward slashes (as_posix())
            assert "data/train/train.parquet" in content
            assert "data/test/test.parquet" in content

    def test_write_config_is_valid_python(self):
        """Test that generated config file is valid Python code.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "config.py"
        ...     file_manager = MagicMock()
        ...     file_manager.train_file_relative_path = Path("data/train.parquet")
        ...     file_manager.test_file_relative_path = Path("data/test.parquet")
        ...     writer = ConfigWriter(str(output_path), "pyarrow", file_manager)
        ...     writer.write_config()
        ...     # Try to compile the generated code
        ...     content = output_path.read_text()
        ...     compile(content, str(output_path), 'exec')
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.py"
            file_manager = MagicMock()
            file_manager.train_file_relative_path = Path("data/train.parquet")
            file_manager.test_file_relative_path = Path("data/test.parquet")
            
            writer = ConfigWriter(
                str(output_path),
                "pyarrow",
                file_manager
            )
            writer.write_config()
            
            # Verify the file can be compiled as Python
            content = output_path.read_text()
            compile(content, str(output_path), 'exec')

    def test_write_config_with_path_object(self):
        """Test write_config accepts Path objects as output_path.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.coding.write_config import ConfigWriter
        >>> from unittest.mock import MagicMock
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     output_path = Path(tmp_dir) / "config.py"
        ...     file_manager = MagicMock()
        ...     file_manager.train_file_relative_path = Path("data/train.parquet")
        ...     file_manager.test_file_relative_path = Path("data/test.parquet")
        ...     writer = ConfigWriter(output_path, "pyarrow", file_manager)
        ...     writer.write_config()
        ...     assert output_path.exists()
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.py"
            file_manager = MagicMock()
            file_manager.train_file_relative_path = Path("data/train.parquet")
            file_manager.test_file_relative_path = Path("data/test.parquet")
            
            writer = ConfigWriter(
                output_path,  # Path object
                "pyarrow",
                file_manager
            )
            writer.write_config()
            
            assert output_path.exists()
