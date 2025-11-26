from bukka.utils.files import file_manager
import pyarrow.parquet as pq
from bukka.utils.bukka_logger import BukkaLogger

logger = BukkaLogger(__name__)

# Class to handle stats
class DatasetStatistics:
    def backend(self):
        import bukka.data_management.wrapper.polars as polars_wrapper
        return polars_wrapper.PolarsOperations()

    def identify_multicollinearity(self):
        # Placeholder for multicollinearity identification logic
        raise NotImplementedError("Multicollinearity identification not implemented.")

class Dataset(DatasetStatistics):
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
        logger.debug(f"Initializing Dataset with target_column='{target_column}', backend='{dataframe_backend}', train_size={train_size}, stratify={stratify}")
        self.file_manager = file_manager
        self.target_column = target_column

        logger.debug("Setting up dataframe backend")
        self._set_backend(dataframe_backend)

        # If a source dataset was copied into the project by FileManager,
        # attempt to load it into the backend so it can be split. The
        # backend is expected to expose a `load_dataset(path)` method; if
        # it does not, skip loading and assume the backend will manage data
        # itself (this keeps unit tests that monkeypatch the backend working).
        dataset_path = getattr(self.file_manager, 'dataset_path', None)
        if dataset_path is not None and dataset_path.exists():
            logger.debug(f"Loading dataset from: {dataset_path}")
            load_fn = getattr(self.backend, 'load_dataset', None)
            if callable(load_fn):
                load_fn(dataset_path)
            else:
                logger.debug("Backend does not have load_dataset method, skipping dataset loading")
        else:
            logger.debug("No dataset path found or dataset does not exist, skipping dataset loading")

        # Always ask the backend to split and write train/test as Parquet files.
        logger.debug(f"Splitting dataset into train/test with train_size={train_size}")
        if strata is None and target_column is None:
            stratify = False
            strata = []
        if stratify is None:
            strata = []

        self.backend.split_dataset(
            train_path=self.file_manager.train_data_file,
            test_path=self.file_manager.test_data_file,
            target_column=target_column,
            strata=strata,
            train_size=train_size,
            stratify=stratify
        )
        logger.debug("Dataset split completed")

        if feature_columns == None:
            logger.debug("Auto-detecting feature columns from training data")
            self.feature_columns = self.backend.get_column_names()
            if target_column:
                self.feature_columns.remove(target_column)
            logger.debug(f"Detected {len(self.feature_columns)} feature columns")
        
        else:
            logger.debug(f"Using provided feature columns: {len(feature_columns)} columns")
            self.feature_columns = feature_columns

        logger.debug(f"Reading schema from: {self.file_manager.train_data_file}")
        schema = pq.read_schema(self.file_manager.train_data_file)
        # Convert pyarrow.Schema to a plain dict of column_name -> pyarrow.DataType
        self.data_schema = {field.name: field.type for field in schema}
        logger.debug(f"Schema loaded with {len(self.data_schema)} columns")
        logger.debug("Dataset initialization complete")

    def _set_backend(self, dataframe_backend):
        """
        Sets the backend for dataframe operations.
        Parameters:
            dataframe_backend (str): The name of the dataframe backend to use. 
                Currently, only 'polars' is supported.
        Raises:
            NotImplementedError: If the specified backend is not supported.
        """
        logger.debug(f"Setting backend to: {dataframe_backend}")
        if dataframe_backend == 'polars':
            from bukka.data_management.wrapper.polars import PolarsOperations

            self.backend = PolarsOperations()
            logger.debug("PolarsOperations backend initialized")

        else:
            logger.error(f"Unsupported backend: {dataframe_backend}")
            raise NotImplementedError()
        
    def __repr__(self):
        return f"Dataset(target_column={self.target_column}, feature_columns={self.feature_columns})"
    