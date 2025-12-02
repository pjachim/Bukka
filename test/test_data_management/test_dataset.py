import pytest
from pathlib import Path

import pyarrow as pa


def create_dummy_dataframe(target_column: str):
    """Create a small test DataFrame for split_dataset operations.
    
    Returns a polars DataFrame with a 'feat' column and the specified target column.
    """
    import polars as pl
    return pl.DataFrame({
        'feat': [1, 2, 3, 4, 5, 6, 7, 8],
        target_column: [0, 1, 0, 1, 0, 1, 0, 1]
    })


class TestDataset:
    """Unit tests for `src/bukka/data_management/dataset.py`.

    These tests work with the current Dataset API which uses composition with
    DatasetIO, DatasetManagement, DatasetStatistics, and DatasetQuality objects.
    Tests monkeypatch methods on these composition objects to avoid requiring actual dataset files.
    """

    def test_init_sets_feature_columns_and_data_schema(self, monkeypatch, tmp_path: Path):
        # Prepare a minimal file-manager-like object with explicit file paths
        fm = type('FM', (), {})()
        fm.train_data = tmp_path / 'train'
        fm.test_data = tmp_path / 'test'
        fm.train_data_file = tmp_path / 'train' / 'train.parquet'
        fm.test_data_file = tmp_path / 'test' / 'test.parquet'
        # Create a fake dataset path that exists
        fm.dataset_path = tmp_path / 'source.csv'
        fm.dataset_path.touch()

        # Import Dataset and monkeypatch DatasetIO.load_from_file to return test data
        from bukka.data_management.dataset import Dataset
        from bukka.data_management.dataset_functionality import DatasetIO
        
        test_df = create_dummy_dataframe('target')
        monkeypatch.setattr(DatasetIO, 'load_from_file', lambda self, path: test_df)

        dset = Dataset(target_column='target', file_manager=fm, train_size=0.5)

        # The dataset creates columns ['feat', 'target'] and removes the target from features
        assert dset.feature_columns == ['feat']

        # data_schema should be a dict read from the parquet file
        assert isinstance(dset.data_schema, dict)
        # Ensure both columns appear in the schema
        assert 'feat' in dset.data_schema
        assert 'target' in dset.data_schema

    def test_repr_includes_target_and_features(self, monkeypatch, tmp_path: Path):
        fm = type('FM', (), {})()
        fm.train_data = tmp_path / 'train'
        fm.test_data = tmp_path / 'test'
        fm.train_data_file = tmp_path / 'train' / 'train.parquet'
        fm.test_data_file = tmp_path / 'test' / 'test.parquet'
        fm.dataset_path = tmp_path / 'source.csv'
        fm.dataset_path.touch()

        from bukka.data_management.dataset import Dataset
        from bukka.data_management.dataset_functionality import DatasetIO
        
        test_df = create_dummy_dataframe('target')
        monkeypatch.setattr(DatasetIO, 'load_from_file', lambda self, path: test_df)

        dset = Dataset(target_column='target', file_manager=fm, train_size=0.5)

        s = repr(dset)
        assert 'Dataset(target_column=target' in s
        assert 'feature_columns' in s or 'feat' in s

    def test_feature_columns_provided_explicitly(self, monkeypatch, tmp_path: Path):
        """Test that explicitly provided feature_columns are used instead of auto-detection."""
        fm = type('FM', (), {})()
        fm.train_data = tmp_path / 'train'
        fm.test_data = tmp_path / 'test'
        fm.train_data_file = tmp_path / 'train' / 'train.parquet'
        fm.test_data_file = tmp_path / 'test' / 'test.parquet'
        fm.dataset_path = tmp_path / 'source.csv'
        fm.dataset_path.touch()

        from bukka.data_management.dataset import Dataset
        from bukka.data_management.dataset_functionality import DatasetIO
        
        test_df = create_dummy_dataframe('target')
        monkeypatch.setattr(DatasetIO, 'load_from_file', lambda self, path: test_df)

        # Provide explicit feature columns
        dset = Dataset(
            target_column='target',
            file_manager=fm,
            train_size=0.5,
            feature_columns=['custom_feat']
        )

        # Should use provided feature columns, not auto-detected
        assert dset.feature_columns == ['custom_feat']
