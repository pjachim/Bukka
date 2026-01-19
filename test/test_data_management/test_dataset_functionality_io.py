"""Unit tests for DatasetIO class."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl
import narwhals as nw

from bukka.data_management.dataset_functionality.io import DatasetIO


class TestDatasetIOInitialization:
    """Test suite for DatasetIO initialization."""

    def test_initialization(self):
        """Test DatasetIO can be instantiated.
        
        Examples
        --------
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> io = DatasetIO()
        >>> assert isinstance(io, DatasetIO)
        """
        io = DatasetIO()
        assert isinstance(io, DatasetIO)


class TestDatasetIOLoadFromCSV:
    """Test suite for DatasetIO.load_from_csv method."""

    def test_load_from_csv_creates_narwhals_dataframe(self):
        """Test load_from_csv returns a Narwhals DataFrame.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        ...     f.write("col1,col2\\n1,2\\n3,4")
        ...     csv_path = f.name
        >>> io = DatasetIO()
        >>> df = io.load_from_csv(csv_path)
        >>> # Clean up
        >>> import os
        >>> os.unlink(csv_path)
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("col1,col2\n1,2\n3,4")
            csv_path = f.name
        
        try:
            io = DatasetIO()
            df = io.load_from_csv(csv_path)
            
            # Verify it's a Narwhals frame (has Narwhals methods)
            assert hasattr(df, 'schema')
            assert len(df) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_from_csv_with_custom_kwargs(self):
        """Test load_from_csv accepts custom polars kwargs.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        ...     f.write("col1;col2\\n1;2\\n3;4")
        ...     csv_path = f.name
        >>> io = DatasetIO()
        >>> df = io.load_from_csv(csv_path, pl_kwargs={'separator': ';'})
        >>> # Clean up
        >>> import os
        >>> os.unlink(csv_path)
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("col1;col2\n1;2\n3;4")
            csv_path = f.name
        
        try:
            io = DatasetIO()
            df = io.load_from_csv(csv_path, pl_kwargs={'separator': ';'})
            
            assert len(df) == 2
            schema = df.schema
            assert 'col1' in schema
            assert 'col2' in schema
        finally:
            Path(csv_path).unlink()

    def test_load_from_csv_reads_correct_data(self):
        """Test load_from_csv reads correct data from CSV.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        ...     f.write("a,b\\n10,20\\n30,40")
        ...     csv_path = f.name
        >>> io = DatasetIO()
        >>> df = io.load_from_csv(csv_path)
        >>> native_df = df.to_native()
        >>> assert native_df['a'].to_list() == [10, 30]
        >>> # Clean up
        >>> import os
        >>> os.unlink(csv_path)
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("a,b\n10,20\n30,40")
            csv_path = f.name
        
        try:
            io = DatasetIO()
            df = io.load_from_csv(csv_path)
            native_df = nw.to_native(df)
            
            assert native_df['a'].to_list() == [10, 30]
            assert native_df['b'].to_list() == [20, 40]
        finally:
            Path(csv_path).unlink()


class TestDatasetIOLoadFromParquet:
    """Test suite for DatasetIO.load_from_parquet method."""

    def test_load_from_parquet_creates_narwhals_dataframe(self):
        """Test load_from_parquet returns a Narwhals DataFrame.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
        ...     parquet_path = f.name
        >>> # Create a parquet file
        >>> df = pl.DataFrame({'x': [1, 2], 'y': [3, 4]})
        >>> df.write_parquet(parquet_path)
        >>> io = DatasetIO()
        >>> loaded_df = io.load_from_parquet(parquet_path)
        >>> # Clean up
        >>> import os
        >>> os.unlink(parquet_path)
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
            parquet_path = f.name
        
        # Create a parquet file
        df = pl.DataFrame({'x': [1, 2], 'y': [3, 4]})
        df.write_parquet(parquet_path)
        
        try:
            io = DatasetIO()
            loaded_df = io.load_from_parquet(parquet_path)
            
            assert hasattr(loaded_df, 'schema')
            assert len(loaded_df) == 2
        finally:
            Path(parquet_path).unlink()

    def test_load_from_parquet_reads_correct_data(self):
        """Test load_from_parquet reads correct data from Parquet file.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
        ...     parquet_path = f.name
        >>> df = pl.DataFrame({'col1': [100, 200], 'col2': [300, 400]})
        >>> df.write_parquet(parquet_path)
        >>> io = DatasetIO()
        >>> loaded_df = io.load_from_parquet(parquet_path)
        >>> native = loaded_df.to_native()
        >>> assert native['col1'].to_list() == [100, 200]
        >>> # Clean up
        >>> import os
        >>> os.unlink(parquet_path)
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
            parquet_path = f.name
        
        df = pl.DataFrame({'col1': [100, 200], 'col2': [300, 400]})
        df.write_parquet(parquet_path)
        
        try:
            io = DatasetIO()
            loaded_df = io.load_from_parquet(parquet_path)
            native = nw.to_native(loaded_df)
            
            assert native['col1'].to_list() == [100, 200]
            assert native['col2'].to_list() == [300, 400]
        finally:
            Path(parquet_path).unlink()


class TestDatasetIOLoadFromFile:
    """Test suite for DatasetIO.load_from_file method."""

    def test_load_from_file_csv(self):
        """Test load_from_file with CSV file type.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        ...     f.write("x,y\\n1,2")
        ...     csv_path = f.name
        >>> io = DatasetIO()
        >>> df = io.load_from_file(csv_path, file_type='csv')
        >>> assert len(df) == 1
        >>> # Clean up
        >>> import os
        >>> os.unlink(csv_path)
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("x,y\n1,2")
            csv_path = f.name
        
        try:
            io = DatasetIO()
            df = io.load_from_file(csv_path, file_type='csv')
            assert len(df) == 1
        finally:
            Path(csv_path).unlink()

    def test_load_from_file_parquet(self):
        """Test load_from_file with Parquet file type.
        
        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
        ...     parquet_path = f.name
        >>> df = pl.DataFrame({'a': [5, 10]})
        >>> df.write_parquet(parquet_path)
        >>> io = DatasetIO()
        >>> loaded = io.load_from_file(parquet_path, file_type='parquet')
        >>> assert len(loaded) == 2
        >>> # Clean up
        >>> import os
        >>> os.unlink(parquet_path)
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
            parquet_path = f.name
        
        df = pl.DataFrame({'a': [5, 10]})
        df.write_parquet(parquet_path)
        
        try:
            io = DatasetIO()
            loaded = io.load_from_file(parquet_path, file_type='parquet')
            assert len(loaded) == 2
        finally:
            Path(parquet_path).unlink()

    def test_load_from_file_raises_on_unsupported_type(self):
        """Test load_from_file raises ValueError for unsupported file types.
        
        Examples
        --------
        >>> from bukka.data_management.dataset_functionality.io import DatasetIO
        >>> io = DatasetIO()
        >>> try:
        ...     io.load_from_file('data.txt', file_type='txt')
        ... except ValueError as e:
        ...     assert 'unsupported' in str(e).lower()
        """
        io = DatasetIO()
        
        with pytest.raises(ValueError) as exc_info:
            io.load_from_file('data.txt', file_type='txt')
        
        assert 'unsupported' in str(exc_info.value).lower()
