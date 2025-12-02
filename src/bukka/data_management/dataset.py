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

class Dataset:
    """Dataset class for managing and splitting datasets for expert systems.
    
    This class loads, splits, and manages datasets, delegating backend operations
    to pluggable implementations (default: Polars). It writes train/test splits as
    Parquet files and exposes schema and feature metadata.
    
    Parameters
    ----------
    target_column : str
        The name of the target column in the dataset.
    file_manager : file_manager.FileManager
        An instance of FileManager to handle file paths and dataset storage.
    strata : list[str] | str | None, optional
        Column(s) to use for stratified splitting. Defaults to None.
    stratify : bool, optional
        Whether to stratify the split. Defaults to True.
    train_size : float, optional
        Proportion of the dataset to include in the train split. Defaults to 0.8.
    feature_columns : list[str] | None, optional
        List of feature column names. If None, all columns except the target are used.
        Defaults to None.
    
    Attributes
    ----------
    file_manager : file_manager.FileManager
        File manager instance for data paths.
    target_column : str
        Name of the target column.
    feature_columns : list[str]
        List of feature column names.
    data_schema : dict[str, pyarrow.DataType]
        Schema of the training data (column names mapped to PyArrow data types).
    train_df : polars.DataFrame
        Training data split.
    test_df : polars.DataFrame
        Test data split.
    
    Examples
    --------
    >>> from bukka.utils.files.file_manager import FileManager
    >>> from pathlib import Path
    >>> fm = FileManager(project_name='my_project', dataset_path=Path('data.csv'))
    >>> dataset = Dataset(
    ...     target_column='label',
    ...     file_manager=fm,
    ...     train_size=0.7,
    ...     stratify=True
    ... )
    >>> print(dataset.feature_columns)
    ['feat1', 'feat2', 'feat3']
    >>> print(dataset.data_schema)
    {'feat1': int64, 'feat2': double, 'label': int64}
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
        self.io = DatasetIO()
        self.management = DatasetManagement()
        self.statistics = DatasetStatistics()
        self.quality = DatasetQuality()

        logger.debug(f"Initializing Dataset with target_column='{target_column}', train_size={train_size}, stratify={stratify}")
        self.file_manager = file_manager
        self.target_column = target_column

        dataset_path = getattr(self.file_manager, 'dataset_path', None)
        if dataset_path is not None and dataset_path.exists():
            logger.debug(f"Loading dataset from: {dataset_path}")
            df = self.io.load_from_file(dataset_path)
        else:
            logger.debug("No dataset path found or dataset does not exist, skipping dataset loading")

        logger.debug(f"Splitting dataset into train/test with train_size={train_size}")
        if strata is None and target_column is None:
            stratify = False
            strata = []
        if stratify is None:
            strata = []

        self.train_df, self.test_df = self.management.split_dataset(
            df=df,
            target_column=target_column,
            strata=strata,
            train_size=train_size,
            stratify=stratify
        )
        self.io.save_to_parquet(self.train_df, self.file_manager.train_data_file)
        self.io.save_to_parquet(self.test_df, self.file_manager.test_data_file)

        logger.debug("Dataset split completed")

        if feature_columns == None:
            logger.debug("Auto-detecting feature columns from training data")
            self.feature_columns = list(self.train_df.columns)
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
        
        Computes pairwise correlations between numerical columns and returns
        pairs with absolute correlation above the specified threshold.
        
        Parameters
        ----------
        columns : list[str], optional
            List of column names to check for multicollinearity. If None,
            uses all feature columns. Defaults to None.
        threshold : float, optional
            Correlation threshold above which column pairs are considered
            multicollinear. Defaults to 0.8.
        
        Returns
        -------
        list[tuple[str, str, float]]
            List of tuples with correlated column pairs and their correlation
            coefficient. Each tuple is (col1, col2, correlation).
        
        Examples
        --------
        >>> dataset = Dataset(
        ...     target_column='label',
        ...     file_manager=fm
        ... )
        >>> pairs = dataset.identify_multicollinearity_train(['feat1', 'feat2', 'feat3'])
        >>> print(pairs)
        [('feat1', 'feat2', 0.95), ('feat2', 'feat3', 0.87)]
        
        >>> # Use default feature columns and custom threshold
        >>> pairs = dataset.identify_multicollinearity_train(threshold=0.9)
        """
        if columns is None:
            columns = self.feature_columns

        return self.statistics.identify_multicollinearity(self.train_df, columns, threshold)
    
    def get_varied_scale_train(self, column_name: str):
        """Calculate the range of a column in the training dataset.
        
        Computes the range (maximum value minus minimum value) for a numerical
        column to assess scale variation.
        
        Parameters
        ----------
        column_name : str
            Name of the column to analyze. Must be a numerical column.
        
        Returns
        -------
        float
            The range of the column (max - min).
        
        Examples
        --------
        >>> dataset = Dataset(
        ...     target_column='label',
        ...     file_manager=fm
        ... )
        >>> scale = dataset.get_varied_scale_train('price')
        >>> print(scale)
        950.75
        
        >>> # Check scale for multiple columns
        >>> for col in ['price', 'quantity', 'weight']:
        ...     scale = dataset.get_varied_scale_train(col)
        ...     print(f"{col}: {scale}")
        """
        return self.statistics.get_varied_scale(self.train_df, column_name)
    
    def check_varied_scale_train(self, column_name: str, threshold: float):
        """Check if a column has varied scale in the training dataset.
        
        Determines whether a numerical column's range exceeds a specified
        threshold, indicating that scaling might be beneficial.
        
        Parameters
        ----------
        column_name : str
            Name of the column to check. Must be a numerical column.
        threshold : float
            The threshold value for determining varied scale. If the column's
            range (max - min) exceeds this value, returns True.
        
        Returns
        -------
        bool
            True if the column's range exceeds the threshold, False otherwise.
        
        Examples
        --------
        >>> dataset = Dataset(
        ...     target_column='label',
        ...     file_manager=fm
        ... )
        >>> has_varied = dataset.check_varied_scale_train('price', 100)
        >>> print(has_varied)
        True
        
        >>> # Check multiple columns for scaling needs
        >>> for col in ['price', 'age', 'quantity']:
        ...     needs_scaling = dataset.check_varied_scale_train(col, threshold=100)
        ...     if needs_scaling:
        ...         print(f"{col} needs scaling")
        """
        return self.statistics.does_data_have_varied_scale(self.train_df, column_name, threshold)
        
    def __repr__(self):
        return f"Dataset(target_column={self.target_column}, feature_columns={self.feature_columns})"
    