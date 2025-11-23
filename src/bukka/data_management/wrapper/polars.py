import polars as pl
from pathlib import Path
import math, random
from typing import Optional, Union, List, Dict, Tuple
from bukka.utils.bukka_logger import BukkaLogger

logger = BukkaLogger(__name__)

class PolarsOperations:
    """
    A utility class for performing common data loading, splitting, and
    descriptive statistics operations using the Polars DataFrame library.

    The class manages two main DataFrame attributes: `train_df` (intended for
    analysis/training) and `full_df` (intended for splitting).

    Attributes
    ----------
    train_df : Optional[pl.DataFrame]
        The primary Polars DataFrame for operations like statistics calculation.
        Initialized to None.
    full_df : Optional[pl.DataFrame]
        A Polars DataFrame, primarily used as the source for the split_dataset method.
        Initialized to None.
    """
    def __init__(self, train_df: Optional[pl.DataFrame] = None, full_df: Optional[pl.DataFrame] = None, random_seed: int | None = None):
        """
        Initializes the PolarsOperations object.

        Parameters
        ----------
        train_df : Optional[pl.DataFrame], default=None
            Initial DataFrame to set as the primary training/analysis DataFrame.
        full_df : Optional[pl.DataFrame], default=None
            Initial DataFrame to set as the full, unsplit DataFrame.
        """
        logger.debug(f"Initializing PolarsOperations with random_seed={random_seed}")
        self.train_df: Optional[pl.DataFrame] = train_df
        self.full_df: Optional[pl.DataFrame] = full_df
        self.orig_random_seed = random_seed
        self.current_seed = random_seed
        logger.debug(f"PolarsOperations initialized: train_df={'set' if train_df is not None else 'None'}, full_df={'set' if full_df is not None else 'None'}")

    # --- Data Loading Methods ---

    def read_csv(self, path: Union[str, Path]) -> None:
        """
        Reads a CSV file from the specified path and sets it as the train_df.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to the CSV file.
        """
        logger.debug(f"Reading CSV file from: {path}")
        # Load the CSV file into a Polars DataFrame
        self.train_df = pl.read_csv(path)
        logger.debug(f"CSV loaded: {self.train_df.height} rows, {len(self.train_df.columns)} columns")

    def read_parquet(self, path: Union[str, Path]) -> None:
        """
        Reads a Parquet file from the specified path and sets it as the train_df.

        Parameters
          ----------
        path : Union[str, Path]
            The file path to the Parquet file.
        """
        logger.debug(f"Reading Parquet file from: {path}")
        # Load the Parquet file into a Polars DataFrame
        self.train_df = pl.read_parquet(path)
        logger.debug(f"Parquet loaded: {self.train_df.height} rows, {len(self.train_df.columns)} columns")

    def load_dataset(self, path: Union[str, Path]) -> None:
        """
        Load a dataset file into `self.full_df`, supporting multiple input formats.

        Supported formats: CSV (.csv), Parquet (.parquet), JSON (.json),
        JSONL/NDJSON (.jsonl, .ndjson).

        Parameters
        ----------
        path : Union[str, Path]
            Path to the source dataset file.
        """
        logger.debug(f"Loading dataset from: {path}")
        p = Path(path)
        suffix = p.suffix.lower()
        logger.debug(f"Detected file format: {suffix}")

        if suffix == '.csv':
            self.full_df = pl.read_csv(p)
        elif suffix in ('.parquet', '.parq', '.pqt'):
            self.full_df = pl.read_parquet(p)
        elif suffix == '.json':
            # Polars can read JSON; for large files consider ndjson
            try:
                self.full_df = pl.read_json(p)
            except Exception:
                # Fallback: try reading as newline-delimited JSON
                self.full_df = pl.read_ndjson(p)
        elif suffix in ('.jsonl', '.ndjson'):
            self.full_df = pl.read_ndjson(p)
        else:
            # Unknown format — raise a helpful error rather than silently failing
            logger.error(f"Unsupported dataset format: {suffix}")
            raise ValueError(f"Unsupported dataset format: {suffix}")
        
        logger.debug(f"Dataset loaded: {self.full_df.height} rows, {len(self.full_df.columns)} columns")

    # --- Data Splitting/Saving Methods ---

    def split_dataset(self, train_path: Union[str, Path], test_path: Union[str, Path],
                      target_column: str, train_size: float = 0.8,
                      stratify: bool = True, strata: Optional[List[str]] = None) -> None:
        """
        Splits the `full_df` into training and testing sets, optionally using stratification,
        and saves them as Parquet files.

        Requires `self.full_df` to be set. The split is performed after shuffling the data
        within each stratum for a randomized split, ensuring reproducibility via the `seed`.

        Parameters
        ----------
        train_path : Union[str, Path]
            The file path to save the training DataFrame as a Parquet file.
        test_path : Union[str, Path]
            The file path to save the testing DataFrame as a Parquet file.
        target_column : str
            The name of the column to use as the target for stratified splitting.
        train_size : float, default=0.8
            The proportion of the dataset to include in the train split (0.0 to 1.0).
        stratify : bool, default=True
            If True, performs stratification based on the specified strata (or `target_column`).
        strata : Optional[List[str]], default=None
            List of column names to use for partitioning/stratification.

        Raises
        ------
        AttributeError
            If `self.full_df` is None.
        """
        logger.debug(f"Starting dataset split: train_size={train_size}, stratify={stratify}, strata={strata}")
        # Ensure that the full_df exists before attempting to split
        if self.full_df is None:
            logger.error("Attempted to split dataset but self.full_df is None")
            raise AttributeError("self.full_df is None. Cannot split the dataset.")

        if stratify:
            # Initialize lists to hold split dataframes for concatenation
            train_results: List[pl.DataFrame] = []
            test_results: List[pl.DataFrame] = []

            # Determine the columns to use for stratification
            strata_cols = strata if strata is not None else [target_column]
            logger.debug(f"Performing stratified split on columns: {strata_cols}")

            # Partition the full DataFrame by the stratification columns
            partition_count = 0
            for df in self.full_df.partition_by(strata_cols):
                partition_count += 1
                logger.debug(f"Processing stratum {partition_count}: {df.height} rows")
                # Split each stratum DataFrame independently, using the seed for shuffling
                train, test = self._split_dataset(df, train_size=train_size, seed=self._random_seed())
                train_results.append(train)
                test_results.append(test)
            logger.debug(f"Processed {partition_count} strata")

            # Concatenate the split results back into final train and test DataFrames
            train_df = pl.concat(train_results)
            test_df = pl.concat(test_results)
            logger.debug(f"Stratified split complete: train={train_df.height} rows, test={test_df.height} rows")

        else:
            logger.debug("Performing non-stratified split")
            # Perform a simple, non-stratified split on the entire full DataFrame, with shuffling
            train_df, test_df = self._split_dataset(self.full_df, train_size=train_size, seed=self._random_seed())
            logger.debug(f"Non-stratified split complete: train={train_df.height} rows, test={test_df.height} rows")

        # Write the resulting DataFrames to disk as Parquet files
        logger.debug(f"Writing train dataset to: {train_path}")
        self.write_parquet(train_df.sample(fraction=1.0, with_replacement=False, seed=self._random_seed()), train_path)
        logger.debug(f"Writing test dataset to: {test_path}")
        self.write_parquet(test_df.sample(fraction=1.0, with_replacement=False, seed=self._random_seed()), test_path)
        logger.debug("Dataset split and save complete")


    def write_parquet(self, df: pl.DataFrame, path: Union[str, Path]) -> None:
        """
        Writes a Polars DataFrame to the specified path in Parquet format.

        Parameters
        ----------
        df : pl.DataFrame
            The Polars DataFrame to be written.
        path : Union[str, Path]
            The file path where the Parquet file will be saved.
        """
        logger.debug(f"Writing {df.height} rows to parquet: {path}")
        # Use Polars' built-in method to write the DataFrame
        df.write_parquet(path)


    def _split_dataset(self, df: pl.DataFrame, train_size: float, seed: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Helper method to perform a reproducible, randomized train/test split on a single DataFrame.

        The DataFrame is shuffled using the provided `seed` before slicing.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to be split.
        train_size : float
            The proportion of the DataFrame to include in the train split (0.0 to 1.0).
        seed : int
            The random seed for shuffling.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the training DataFrame and the testing DataFrame.
        """
        logger.debug(f"Splitting DataFrame of {df.height} rows with train_size={train_size}, seed={seed}")
        # 1. Shuffle the DataFrame for a randomized split
        shuffled_df: pl.DataFrame = df.sample(fraction=1.0, with_replacement=False, seed=seed)
        
        # 2. Calculate the split point
        # Calculate the number of samples for the training set, rounding up.
        n_train: int = math.ceil(shuffled_df.height * train_size)

        # The test size is the total height minus the calculated train size.
        n_test: int = shuffled_df.height - n_train

        # 3. Slice the shuffled DataFrame
        train_df = shuffled_df.slice(0, n_train)
        test_df = shuffled_df.slice(n_train, n_test)
        logger.debug(f"Split result: train={n_train} rows, test={n_test} rows")

        return train_df, test_df
    
    def _random_seed(self):
        """
        Provide a seed for the partition. This creates a different seed each time to better shuffle across partitions.
        """
        random.seed(self.current_seed)
        self.current_seed = random.randint(1, 1_000_000)
        logger.debug(f"Generated new random seed: {self.current_seed}")
        return self.current_seed

    # --- Descriptive Statistics / Metadata Methods ---
    # (Statistical methods remain the same as they operate only on train_df)

    def get_column_names(self) -> List[str]:
        """
        Retrieves the names of all columns in `self.train_df`.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        List[str]
            A list of column names.
        """
        self._ensure_df()
        return self.train_df.columns

    def get_dtypes(self) -> Dict[str, pl.DataType]:
        """
        Retrieves the data types (schema) of all columns in `self.train_df`.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        Dict[str, pl.DataType]
            A dictionary where keys are column names and values are Polars data types.
        """
        self._ensure_df()
        return dict(self.train_df.schema)

    def get_column_mean(self, column: str) -> float:
        """
        Calculates the mean (average) of a specified column in `self.train_df`.

        Parameters
        ----------
        column : str
            The name of the column.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        float
            The mean value of the column.
        """
        self._ensure_df()
        return self.train_df.select(pl.col(column).mean()).item()

    def get_column_std(self, column: str) -> float:
        """
        Calculates the standard deviation of a specified column in `self.train_df`.

        Parameters
        ----------
        column : str
            The name of the column.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        float
            The standard deviation of the column.
        """
        self._ensure_df()
        return self.train_df.select(pl.col(column).std()).item()

    def get_column_median(self, column: str) -> float:
        """
        Calculates the median (50th percentile) of a specified column in `self.train_df`.

        Parameters
        ----------
        column : str
            The name of the column.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        float
            The median value of the column.
        """
        self._ensure_df()
        return self.train_df.select(pl.col(column).median()).item()

    def get_column_null_count(self, column: str) -> int:
        """
        Counts the number of null (missing) values in a specified column in `self.train_df`.

        Parameters
        ----------
        column : str
            The name of the column.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        int
            The count of null values in the column.
        """
        self._ensure_df()
        return self.train_df.select(pl.col(column).is_null().sum()).item()

    def get_number_of_samples(self) -> int:
        """
        Retrieves the number of rows (height) in `self.train_df`.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        int
            The number of rows in the DataFrame.
        """
        self._ensure_df()
        return self.train_df.height
    
    def get_unq_count(self, column: str) -> int:
        """
        Retrieves the number of unique values in a specified column in `self.train_df`.

        Parameters
        ----------
        column : str
            The name of the column.

        Raises
        ------
        AttributeError
            If `self.train_df` is None.

        Returns
        -------
        int
            The count of unique values in the column.
        """
        self._ensure_df()
        return self.train_df.select(pl.col(column).n_unique()).item()

    def _ensure_df(self) -> None:
        """
        Private helper method to ensure that the `self.train_df` attribute is set (not None).

        Raises
        ------
        AttributeError
            If `self.train_df` is None, indicating a data loading method was not called.
        """
        if self.train_df is None:
            logger.error("Attempted to access train_df but it is None")
            # Raise an informative error if the DataFrame is not loaded
            raise AttributeError('self.train_df is None. Call a method with the prefix read_ (e.g., read_csv), and try again.')