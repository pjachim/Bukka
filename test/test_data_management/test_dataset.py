import pytest
from pathlib import Path

import pyarrow as pa


class DummyPolarsOps:
    """A lightweight stand-in for the real PolarsOperations used in tests.

    It writes small parquet files for the train/test split and exposes a
    `get_column_names` method via `train_df` (the production code expects
    `backend.train_df.get_column_names()`).
    """
    def __init__(self):
        self.train_df = None
        self._columns = []

    def split_dataset(self, train_path: Path, test_path: Path, target_column: str, strata=None, train_size: float = 0.8, stratify: bool = True):
        import polars as pl

        # Create a tiny DataFrame with a feature and the target column
        df = pl.DataFrame({
            'feat': [1, 2, 3, 4],
            target_column: [0, 1, 0, 1]
        })

        n_train = int(len(df) * train_size)
        # Ensure at least zero rows work
        train_df = df.head(n_train)
        test_df = df.tail(len(df) - n_train)

        # Write parquet files so pyarrow can read the schema later
        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)

        # Expose get_column_names via train_df as the production code expects
        self.train_df = self
        self._columns = list(train_df.columns)

    def get_column_names(self):
        return self._columns


class TestDataset:
    """Unit tests for `src/bukka/data_management/dataset.py`.

    These tests monkeypatch the Polars backend used by `Dataset` so that
    construction does not depend on external state and writes predictable
    parquet files for schema inspection.
    """

    def test_init_sets_feature_columns_and_data_schema(self, monkeypatch, tmp_path: Path):
        # Prepare a minimal file-manager-like object with explicit file paths
        fm = type('FM', (), {})()
        fm.train_data = tmp_path / 'train.parquet'
        fm.test_data = tmp_path / 'test.parquet'

        # Monkeypatch the PolarsOperations class used by Dataset to our dummy
        import bukka.data_management.wrapper.polars as ds_module

        monkeypatch.setattr(ds_module, 'PolarsOperations', DummyPolarsOps)

        # Import Dataset lazily from the module under test
        from bukka.data_management.dataset import Dataset

        dset = Dataset(target_column='target', file_manager=fm, dataframe_backend='polars', train_size=0.5)

        # The DummyPolarsOps creates columns ['feat', 'target'] and removes the target
        assert dset.feature_columns == ['feat']

        # data_schema should be a dict read from the parquet file written by the dummy backend
        assert isinstance(dset.data_schema, dict)
        # Ensure both columns appear in the schema
        assert 'feat' in dset.data_schema
        assert 'target' in dset.data_schema

    def test_repr_includes_target_and_features(self, monkeypatch, tmp_path: Path):
        fm = type('FM', (), {})()
        fm.train_data = tmp_path / 'train.parquet'
        fm.test_data = tmp_path / 'test.parquet'

        import bukka.data_management.wrapper.polars as ds_module
        monkeypatch.setattr(ds_module, 'PolarsOperations', DummyPolarsOps)
        from bukka.data_management.dataset import Dataset

        dset = Dataset(target_column='target', file_manager=fm, dataframe_backend='polars', train_size=0.5)

        s = repr(dset)
        assert 'Dataset(target_column=target' in s
        assert 'feature_columns' in s or 'feat' in s

    def test_set_backend_raises_for_unknown_backend(self, monkeypatch, tmp_path: Path):
        # Set up file manager paths
        fm = type('FM', (), {})()
        fm.train_data = tmp_path / 'train.parquet'
        fm.test_data = tmp_path / 'test.parquet'

        import bukka.data_management.wrapper.polars as ds_module
        monkeypatch.setattr(ds_module, 'PolarsOperations', DummyPolarsOps)
        from bukka.data_management.dataset import Dataset

        dset = Dataset(target_column='target', file_manager=fm, dataframe_backend='polars', train_size=0.5)

        # Attempt to set an unsupported backend
        with pytest.raises(NotImplementedError):
            dset._set_backend('not_a_real_backend')
