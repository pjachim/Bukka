from bukka.utils.files import file_manager
import pyarrow.parquet as pq
from bukka.utils.bukka_logger import BukkaLogger
from bukka.data_management.dataset_functionality import (
    DatasetStatistics,
    DatasetManagement,
    DatasetIO,
    DatasetQuality,
)
logger = BukkaLogger(__name__)

class Dataset(DatasetStatistics, DatasetManagement, DatasetIO, DatasetQuality):
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
            strata=None,
            stratify=True,
            train_size=0.8,
            feature_columns: list[str] | None = None
        ):
        logger.debug(f"Initializing Dataset with target_column='{target_column}', train_size={train_size}, stratify={stratify}")
        self.file_manager = file_manager
        self.target_column = target_column

        # If a source dataset was copied into the project by FileManager,
        # attempt to load it into the backend so it can be split. The
        # backend is expected to expose a `load_dataset(path)` method; if
        # it does not, skip loading and assume the backend will manage data
        # itself (this keeps unit tests that monkeypatch the backend working).
        dataset_path = getattr(self.file_manager, 'dataset_path', None)
        if dataset_path is not None and dataset_path.exists():
            logger.debug(f"Loading dataset from: {dataset_path}")
            df = self.load_from_file(dataset_path)
        else:
            logger.debug("No dataset path found or dataset does not exist, skipping dataset loading")

        # Always ask the backend to split and write train/test as Parquet files.
        logger.debug(f"Splitting dataset into train/test with train_size={train_size}")
        if strata is None and target_column is None:
            stratify = False
            strata = []
        if stratify is None:
            strata = []

        self.train_df, self.test_df = self.split_dataset(
            df=df,
            target_column=target_column,
            strata=strata,
            train_size=train_size,
            stratify=stratify
        )
        self.save_to_parquet(self.train_df, self.file_manager.train_data_file)
        self.save_to_parquet(self.test_df, self.file_manager.test_data_file)

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
    
    def identify_multicollinearity_train(self, columns: list[str] = None, threshold: float = 0.8):
        """Identify multicollinear features in the training dataset.
        
        Parameters
        ----------
        columns : list[str]
            List of column names to check for multicollinearity.
        threshold : float, optional
            Correlation threshold, by default 0.8.
        
        Returns
        -------
        list[tuple[str, str, float]]
            List of tuples with correlated column pairs and their correlation.
        
        Examples
        --------
        >>> dataset = Dataset(...)
        >>> pairs = dataset.identify_multicollinearity_train(['feat1', 'feat2'])
        """
        if columns is None:
            columns = self.feature_columns

        return self.identify_multicollinearity(self.train_df, columns, threshold)
    
    def get_varied_scale_train(self, column_name: str):
        """Calculate the range of a column in the training dataset.
        
        Parameters
        ----------
        column_name : str
            Name of the column to analyze.
        
        Returns
        -------
        float
            The range of the column (max - min).
        
        Examples
        --------
        >>> dataset = Dataset(...)
        >>> scale = dataset.get_varied_scale_train('price')
        """
        return self.varied_scale(self.train_df, column_name)
    
    def check_varied_scale_train(self, column_name: str, threshold: float):
        """Check if a column has varied scale in the training dataset.
        
        Parameters
        ----------
        column_name : str
            Name of the column to check.
        threshold : float
            The threshold value for determining varied scale.
        
        Returns
        -------
        bool
            True if the column's range exceeds the threshold.
        
        Examples
        --------
        >>> dataset = Dataset(...)
        >>> has_varied = dataset.check_varied_scale_train('price', 100)
        """
        return self.does_data_have_varied_scale(self.train_df, column_name, threshold)
        
    def __repr__(self):
        return f"Dataset(target_column={self.target_column}, feature_columns={self.feature_columns})"
    