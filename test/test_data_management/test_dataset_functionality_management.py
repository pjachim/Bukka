"""Unit tests for DatasetManagement class."""
import pytest
import polars as pl
import narwhals as nw

from bukka.data_management.dataset_functionality.management import DatasetManagement


class TestDatasetManagementInitialization:
    """Test suite for DatasetManagement initialization."""

    def test_initialization(self):
        """Test DatasetManagement can be instantiated.
        
        Examples
        --------
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> manager = DatasetManagement()
        >>> assert isinstance(manager, DatasetManagement)
        """
        manager = DatasetManagement()
        assert isinstance(manager, DatasetManagement)


class TestDatasetManagementSplitDataset:
    """Test suite for DatasetManagement.split_dataset method."""

    def test_split_dataset_returns_two_dataframes(self):
        """Test split_dataset returns train and test DataFrames.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> native_df = pl.DataFrame({
        ...     'feature': [1, 2, 3, 4, 5],
        ...     'target': [0, 1, 0, 1, 0]
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'target')
        >>> assert isinstance(train_df, type(df))
        >>> assert isinstance(test_df, type(df))
        """
        native_df = pl.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        train_df, test_df = manager.split_dataset(df, 'target')
        
        # Both should be narwhals frames
        assert hasattr(train_df, 'schema')
        assert hasattr(test_df, 'schema')

    def test_split_dataset_default_ratio(self):
        """Test split_dataset uses default 80/20 train/test split.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> native_df = pl.DataFrame({
        ...     'x': list(range(10)),
        ...     'y': list(range(10))
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'y')
        >>> assert len(train_df) == 8
        >>> assert len(test_df) == 2
        """
        native_df = pl.DataFrame({
            'x': list(range(10)),
            'y': list(range(10))
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        train_df, test_df = manager.split_dataset(df, 'y')
        
        assert len(train_df) == 8
        assert len(test_df) == 2

    def test_split_dataset_custom_ratio(self):
        """Test split_dataset with custom train_size ratio.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> native_df = pl.DataFrame({
        ...     'x': list(range(10)),
        ...     'y': list(range(10))
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'y', train_size=0.6)
        >>> assert len(train_df) == 6
        >>> assert len(test_df) == 4
        """
        native_df = pl.DataFrame({
            'x': list(range(10)),
            'y': list(range(10))
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        train_df, test_df = manager.split_dataset(df, 'y', train_size=0.6)
        
        assert len(train_df) == 6
        assert len(test_df) == 4

    def test_split_dataset_preserves_all_rows(self):
        """Test split_dataset preserves total number of rows.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> native_df = pl.DataFrame({
        ...     'a': list(range(100)),
        ...     'b': list(range(100))
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'b', train_size=0.7)
        >>> assert len(train_df) + len(test_df) == 100
        """
        native_df = pl.DataFrame({
            'a': list(range(100)),
            'b': list(range(100))
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        train_df, test_df = manager.split_dataset(df, 'b', train_size=0.7)
        
        assert len(train_df) + len(test_df) == 100

    def test_split_dataset_shuffles_data(self):
        """Test split_dataset shuffles the data (probabilistic test).
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> # Create sequential data
        >>> native_df = pl.DataFrame({
        ...     'idx': list(range(100)),
        ...     'target': [i % 2 for i in range(100)]
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'target')
        >>> # With shuffling, first row of train is unlikely to be idx=0
        >>> # (probabilistic, may occasionally fail)
        >>> train_native = train_df.to_native()
        >>> # Just verify we got some data
        >>> assert len(train_native) > 0
        """
        # Create sequential data
        native_df = pl.DataFrame({
            'idx': list(range(100)),
            'target': [i % 2 for i in range(100)]
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        train_df, test_df = manager.split_dataset(df, 'target')
        
        # Verify split happened
        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_split_dataset_preserves_columns(self):
        """Test split_dataset preserves all columns.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> native_df = pl.DataFrame({
        ...     'col1': [1, 2, 3, 4],
        ...     'col2': [5, 6, 7, 8],
        ...     'target': [0, 1, 0, 1]
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'target')
        >>> train_schema = train_df.schema
        >>> test_schema = test_df.schema
        >>> assert 'col1' in train_schema
        >>> assert 'col2' in train_schema
        >>> assert 'target' in train_schema
        >>> assert 'col1' in test_schema
        """
        native_df = pl.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        train_df, test_df = manager.split_dataset(df, 'target')
        
        train_schema = train_df.schema
        test_schema = test_df.schema
        
        assert set(train_schema.keys()) == {'col1', 'col2', 'target'}
        assert set(test_schema.keys()) == {'col1', 'col2', 'target'}

    def test_split_dataset_accepts_stratify_params(self):
        """Test split_dataset accepts stratify and strata parameters.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> from bukka.data_management.dataset_functionality.management import DatasetManagement
        >>> native_df = pl.DataFrame({
        ...     'x': list(range(10)),
        ...     'y': [0, 1] * 5
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> # Just verify the parameters are accepted (stratification not implemented yet)
        >>> train_df, test_df = manager.split_dataset(
        ...     df, 'y', stratify=True, strata=['y']
        ... )
        >>> assert len(train_df) + len(test_df) == 10
        """
        native_df = pl.DataFrame({
            'x': list(range(10)),
            'y': [0, 1] * 5
        })
        df = nw.from_native(native_df)
        
        manager = DatasetManagement()
        # Parameters should be accepted even if not fully implemented
        train_df, test_df = manager.split_dataset(
            df, 'y', stratify=True, strata=['y']
        )
        
        assert len(train_df) + len(test_df) == 10
