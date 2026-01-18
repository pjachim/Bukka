"""Unit tests for DatasetStatistics class."""
import pytest
import polars as pl
import narwhals as nw

from bukka.data_management.dataset_functionality.statistics import DatasetStatistics


class TestDatasetStatisticsInitialization:
    """Test suite for DatasetStatistics initialization."""

    def test_initialization(self):
        """Test DatasetStatistics can be instantiated.
        
        Examples
        --------
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> stats = DatasetStatistics()
        >>> assert isinstance(stats, DatasetStatistics)
        """
        stats = DatasetStatistics()
        assert isinstance(stats, DatasetStatistics)


class TestDatasetStatisticsVariedScale:
    """Test suite for DatasetStatistics.varied_scale method."""

    def test_varied_scale_calculates_range(self):
        """Test varied_scale calculates max - min correctly.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({'values': [1, 5, 10, 100]})
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> scale = stats.varied_scale(df, 'values')
        >>> assert scale == 99
        """
        native_df = pl.DataFrame({'values': [1, 5, 10, 100]})
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        scale = stats.varied_scale(df, 'values')
        assert scale == 99

    def test_varied_scale_zero_range(self):
        """Test varied_scale with all same values (zero range).
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({'values': [5, 5, 5, 5]})
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> scale = stats.varied_scale(df, 'values')
        >>> assert scale == 0
        """
        native_df = pl.DataFrame({'values': [5, 5, 5, 5]})
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        scale = stats.varied_scale(df, 'values')
        assert scale == 0

    def test_varied_scale_negative_values(self):
        """Test varied_scale with negative values.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({'values': [-10, 0, 10]})
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> scale = stats.varied_scale(df, 'values')
        >>> assert scale == 20
        """
        native_df = pl.DataFrame({'values': [-10, 0, 10]})
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        scale = stats.varied_scale(df, 'values')
        assert scale == 20


class TestDatasetStatisticsDoesDataHaveVariedScale:
    """Test suite for DatasetStatistics.does_data_have_varied_scale method."""

    def test_does_data_have_varied_scale_above_threshold(self):
        """Test detection when scale exceeds threshold.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({'values': [0, 100]})
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> # Scale is 100, threshold 50
        >>> assert stats.does_data_have_varied_scale(df, 'values', 50) is True
        """
        native_df = pl.DataFrame({'values': [0, 100]})
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        result = stats.does_data_have_varied_scale(df, 'values', 50)
        assert result == True

    def test_does_data_have_varied_scale_below_threshold(self):
        """Test detection when scale is below threshold.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({'values': [0, 10]})
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> # Scale is 10, threshold 50
        >>> assert stats.does_data_have_varied_scale(df, 'values', 50) is False
        """
        native_df = pl.DataFrame({'values': [0, 10]})
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        result = stats.does_data_have_varied_scale(df, 'values', 50)
        assert result == False


class TestDatasetStatisticsIdentifyMulticollinearity:
    """Test suite for DatasetStatistics.identify_multicollinearity method."""

    def test_identify_multicollinearity_no_correlation(self):
        """Test with uncorrelated columns.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [5, 4, 3, 2, 1]
        ... })
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> pairs = stats.identify_multicollinearity(df, ['a', 'b'], threshold=0.8)
        >>> # These are negatively correlated but abs value should be > 0.8
        """
        native_df = pl.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        pairs = stats.identify_multicollinearity(df, ['a', 'b'], threshold=0.99)
        # Should find the perfect negative correlation
        assert len(pairs) >= 1

    def test_identify_multicollinearity_perfect_correlation(self):
        """Test with perfectly correlated columns.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [2, 4, 6, 8, 10]  # Perfectly correlated (b = 2*a)
        ... })
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> pairs = stats.identify_multicollinearity(df, ['a', 'b'], threshold=0.8)
        >>> assert len(pairs) >= 1
        """
        native_df = pl.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10]  # Perfectly correlated (b = 2*a)
        })
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        pairs = stats.identify_multicollinearity(df, ['a', 'b'], threshold=0.8)
        assert len(pairs) >= 1
        # Should return tuples
        assert isinstance(pairs[0], tuple)
        assert len(pairs[0]) == 3  # (col1, col2, correlation)

    def test_identify_multicollinearity_returns_empty_for_single_column(self):
        """Test with only one column returns empty list.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({'a': [1, 2, 3]})
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> pairs = stats.identify_multicollinearity(df, ['a'], threshold=0.8)
        >>> assert len(pairs) == 0
        """
        native_df = pl.DataFrame({'a': [1, 2, 3]})
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        pairs = stats.identify_multicollinearity(df, ['a'], threshold=0.8)
        assert len(pairs) == 0

    def test_identify_multicollinearity_filters_non_numeric(self):
        """Test that non-numeric columns are filtered out.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.statistics import DatasetStatistics
        >>> native_df = pl.DataFrame({
        ...     'num1': [1, 2, 3],
        ...     'text': ['a', 'b', 'c'],
        ...     'num2': [2, 4, 6]
        ... })
        >>> df = nw.from_native(native_df)
        >>> stats = DatasetStatistics()
        >>> # Should only check num1 and num2, ignore text
        >>> pairs = stats.identify_multicollinearity(df, ['num1', 'text', 'num2'], threshold=0.8)
        >>> # Should find correlation between num1 and num2
        """
        native_df = pl.DataFrame({
            'num1': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'num2': [2, 4, 6]
        })
        df = nw.from_native(native_df)
        stats = DatasetStatistics()
        
        # Should only check num1 and num2, ignore text
        pairs = stats.identify_multicollinearity(df, ['num1', 'text', 'num2'], threshold=0.8)
        # Should find correlation between num1 and num2
        assert len(pairs) >= 1
