"""Unit tests for DatasetQuality class."""
import pytest
import polars as pl
import narwhals as nw

from bukka.data_management.dataset_functionality.quality import DatasetQuality


class TestDatasetQualityInitialization:
    """Test suite for DatasetQuality initialization."""

    def test_initialization(self):
        """Test DatasetQuality can be instantiated.
        
        Examples
        --------
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> quality = DatasetQuality()
        >>> assert isinstance(quality, DatasetQuality)
        """
        quality = DatasetQuality()
        assert isinstance(quality, DatasetQuality)


class TestDatasetQualityGetColumnNullCount:
    """Test suite for DatasetQuality.get_column_null_count method."""

    def test_get_column_null_count_no_nulls(self):
        """Test get_column_null_count with column having no nulls.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'col': [1, 2, 3]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.get_column_null_count(df, 'col') == 0
        """
        native_df = pl.DataFrame({'col': [1, 2, 3]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.get_column_null_count(df, 'col')
        assert result == 0

    def test_get_column_null_count_with_nulls(self):
        """Test get_column_null_count with column having nulls.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'col': [1, None, 3, None, 5]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.get_column_null_count(df, 'col') == 2
        """
        native_df = pl.DataFrame({'col': [1, None, 3, None, 5]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.get_column_null_count(df, 'col')
        assert result == 2

    def test_get_column_null_count_all_nulls(self):
        """Test get_column_null_count with all nulls.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'col': [None, None, None]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.get_column_null_count(df, 'col') == 3
        """
        native_df = pl.DataFrame({'col': [None, None, None]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.get_column_null_count(df, 'col')
        assert result == 3


class TestDatasetQualityTypeOfColumn:
    """Test suite for DatasetQuality.type_of_column method."""

    def test_type_of_column_int(self):
        """Test type_of_column recognizes integer columns.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'col': [1, 2, 3]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.type_of_column(df, 'col') == 'int'
        """
        native_df = pl.DataFrame({'col': [1, 2, 3]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.type_of_column(df, 'col')
        assert result == 'int'

    def test_type_of_column_float(self):
        """Test type_of_column recognizes float columns.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'col': [1.0, 2.5, 3.7]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.type_of_column(df, 'col') == 'float'
        """
        native_df = pl.DataFrame({'col': [1.0, 2.5, 3.7]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.type_of_column(df, 'col')
        assert result == 'float'

    def test_type_of_column_string(self):
        """Test type_of_column recognizes string columns.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'col': ['a', 'b', 'c']})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.type_of_column(df, 'col') == 'string'
        """
        native_df = pl.DataFrame({'col': ['a', 'b', 'c']})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.type_of_column(df, 'col')
        assert result == 'string'


class TestDatasetQualityHasInconsistentCategoricalData:
    """Test suite for DatasetQuality.has_inconsistent_categorical_data method."""

    def test_inconsistent_data_case_variations(self):
        """Test detection of case variations in categorical data.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'cat': ['Cat', 'cat', 'CAT', 'Dog']})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.has_inconsistent_categorical_data(df, 'cat') is True
        """
        native_df = pl.DataFrame({'cat': ['Cat', 'cat', 'CAT', 'Dog']})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.has_inconsistent_categorical_data(df, 'cat')
        assert result is True

    def test_consistent_data_no_issues(self):
        """Test no inconsistency detected in clean categorical data with low unique ratio.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> # Need many rows to keep unique ratio below threshold
        >>> native_df = pl.DataFrame({'cat': ['Cat'] * 50 + ['Dog'] * 50})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> assert quality.has_inconsistent_categorical_data(df, 'cat') is False
        """
        # Create dataset with low unique ratio (2 unique out of 100 = 2%)
        native_df = pl.DataFrame({'cat': ['Cat'] * 50 + ['Dog'] * 50})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.has_inconsistent_categorical_data(df, 'cat')
        assert result is False

    def test_inconsistent_data_high_unique_ratio(self):
        """Test detection of high unique value ratio.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> # Each value is unique (100% unique ratio)
        >>> native_df = pl.DataFrame({'cat': [f'val_{i}' for i in range(100)]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> # Default threshold is 0.1, so 100% unique should trigger
        >>> assert quality.has_inconsistent_categorical_data(df, 'cat') is True
        """
        # Each value is unique (100% unique ratio)
        native_df = pl.DataFrame({'cat': [f'val_{i}' for i in range(100)]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.has_inconsistent_categorical_data(df, 'cat')
        assert result is True


class TestDatasetQualityCheckMissingValues:
    """Test suite for DatasetQuality.check_missing_values method."""

    @pytest.mark.xfail(reason="Known issue: Narwhals DataFrame construction from dict")
    def test_check_missing_values_no_nulls(self):
        """Test check_missing_values with no missing values.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({
        ...     'col1': [1, 2, 3],
        ...     'col2': [4, 5, 6]
        ... })
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> result = quality.check_missing_values(df)
        >>> native_result = result.to_native()
        >>> assert all(native_result['missing_count'] == 0)
        """
        native_df = pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.check_missing_values(df)
        native_result = nw.to_native(result)
        
        assert all(native_result['missing_count'] == 0)

    @pytest.mark.xfail(reason="Known issue: Narwhals DataFrame construction from dict")
    def test_check_missing_values_with_nulls(self):
        """Test check_missing_values with missing values.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({
        ...     'col1': [1, None, 3],
        ...     'col2': [None, 5, None]
        ... })
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> result = quality.check_missing_values(df)
        >>> native_result = result.to_native()
        >>> # col1 should have 1 missing, col2 should have 2 missing
        """
        native_df = pl.DataFrame({
            'col1': [1, None, 3],
            'col2': [None, 5, None]
        })
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.check_missing_values(df)
        native_result = nw.to_native(result)
        
        # Verify structure
        assert 'column' in native_result.columns
        assert 'missing_count' in native_result.columns
        
        # Verify counts
        col1_row = native_result.filter(pl.col('column') == 'col1')
        col2_row = native_result.filter(pl.col('column') == 'col2')
        assert col1_row['missing_count'][0] == 1
        assert col2_row['missing_count'][0] == 2

    @pytest.mark.xfail(reason="Known issue: Narwhals DataFrame construction from dict")
    def test_check_missing_values_returns_narwhals_frame(self):
        """Test check_missing_values returns a Narwhals DataFrame.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.quality import DatasetQuality
        >>> native_df = pl.DataFrame({'x': [1, 2, 3]})
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> result = quality.check_missing_values(df)
        >>> assert hasattr(result, 'schema')
        """
        native_df = pl.DataFrame({'x': [1, 2, 3]})
        df = nw.from_native(native_df)
        quality = DatasetQuality()
        
        result = quality.check_missing_values(df)
        
        # Should be narwhals frame
        assert hasattr(result, 'schema')
