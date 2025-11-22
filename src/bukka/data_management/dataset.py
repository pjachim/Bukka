from bukka.logistics.files import file_manager
import pyarrow
import pyarrow.parquet as pq

class Dataset:
    """
    Dataset class for managing and splitting datasets for expert systems.
    Args:
        target_column (str): The name of the target column in the dataset.
        file_manager (file_manager.FileManager): An instance of FileManager to handle file paths.
        dataframe_backend (str, optional): The backend to use for dataframe operations. Defaults to 'polars'.
        strata (optional): Column(s) to use for stratified splitting. Defaults to None.
        stratify (bool, optional): Whether to stratify the split. Defaults to True.
        train_size (float, optional): Proportion of the dataset to include in the train split. Defaults to 0.8.
        feature_columns (list[str] | None, optional): List of feature column names. If None, all columns except the target are used.
    Attributes:
        file_manager (file_manager.FileManager): File manager instance for data paths.
        target_column (str): Name of the target column.
        backend: Backend operations handler (e.g., PolarsOperations).
        feature_columns (list[str]): List of feature column names.
        data_schema (dict): Schema of the training data.
    Methods:
        _set_backend(dataframe_backend): Sets the backend for dataframe operations.
    Raises:
        NotImplementedError: If a backend other than 'polars' is specified.
    """
    def __init__(
            self,
            target_column: str,
            file_manager: file_manager.FileManager,
            dataframe_backend='polars',
            strata=None,
            stratify=True,
            train_size=0.8,
            feature_columns: list[str] | None = None
        ):
        self.file_manager = file_manager
        self.target_column = target_column

        self._set_backend(dataframe_backend)

        # If a source dataset was copied into the project by FileManager,
        # attempt to load it into the backend so it can be split. The
        # backend is expected to expose a `load_dataset(path)` method; if
        # it does not, skip loading and assume the backend will manage data
        # itself (this keeps unit tests that monkeypatch the backend working).
        dataset_path = getattr(self.file_manager, 'dataset_path', None)
        if dataset_path is not None and dataset_path.exists():
            load_fn = getattr(self.backend, 'load_dataset', None)
            if callable(load_fn):
                load_fn(dataset_path)

        # Always ask the backend to split and write train/test as Parquet files.
        self.backend.split_dataset(
            train_path=self.file_manager.train_data,
            test_path=self.file_manager.test_data,
            target_column=target_column,
            strata=strata,
            train_size=train_size,
            stratify=stratify
        )

        if feature_columns == None:
            self.feature_columns = self.backend.train_df.get_column_names()
            self.feature_columns.remove(target_column)
        
        else:
            self.feature_columns = feature_columns

        schema = pq.read_schema(self.file_manager.train_data)
        # Convert pyarrow.Schema to a plain dict of column_name -> pyarrow.DataType
        self.data_schema = {field.name: field.type for field in schema}

    def _set_backend(self, dataframe_backend):
        """
        Sets the backend for dataframe operations.
        Parameters:
            dataframe_backend (str): The name of the dataframe backend to use. 
                Currently, only 'polars' is supported.
        Raises:
            NotImplementedError: If the specified backend is not supported.
        """
        if dataframe_backend == 'polars':
            from bukka.data_management.wrapper.polars import PolarsOperations

            self.backend = PolarsOperations()

        else:
            raise NotImplementedError()
        
    def __repr__(self):
        return f"Dataset(target_column={self.target_column}, feature_columns={self.feature_columns})"
    