import narwhals as nw
from narwhals.typing import FrameT
import polars as pl
from pathlib import Path
from typing import Any

class DatasetIO:
    """Class to handle dataset input/output operations.
    
    This class provides methods for loading and saving datasets in various
    formats including CSV and Parquet using Narwhals for dataframe abstraction.
    """
    def __init__(self):
        pass

    def load_from_csv(self, file_path: str, pl_kwargs: dict | None = None) -> FrameT:
        """Load dataset from a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file to load.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to the native backend read_csv(),
            by default None.
        
        Returns
        -------
        Narwhals DataFrame
            The loaded DataFrame wrapped in Narwhals.
        
        Examples
        --------
        >>> io = DatasetIO()
        >>> df = io.load_from_csv('data.csv')
        >>> # Load with custom options
        >>> df = io.load_from_csv('data.csv', pl_kwargs={'separator': ';', 'has_header': True})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        # Load using Polars as default backend, then wrap with Narwhals
        native_df = pl.read_csv(file_path, **(pl_kwargs or {}))
        return nw.from_native(native_df)
    
    def load_from_parquet(self, file_path: str, pl_kwargs: dict | None = None) -> FrameT:
        """Load dataset from a Parquet file.
        
        Parameters
        ----------
        file_path : str
            Path to the Parquet file to load.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to the native backend read_parquet(),
            by default None.
        
        Returns
        -------
        Narwhals DataFrame
            The loaded DataFrame wrapped in Narwhals.
        
        Examples
        --------
        >>> io = DatasetIO()
        >>> df = io.load_from_parquet('data.parquet')
        >>> # Load with custom options
        >>> df = io.load_from_parquet('data.parquet', pl_kwargs={'columns': ['col1', 'col2']})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        # Load using Polars as default backend, then wrap with Narwhals
        native_df = pl.read_parquet(file_path, **(pl_kwargs or {}))
        return nw.from_native(native_df)
    
    def load_from_file(self, file_path: str, file_type: str | None = None, pl_kwargs: dict | None = None) -> FrameT:
        """Load dataset from a file based on its type.
        
        Parameters
        ----------
        file_path : str
            Path to the file to load.
        file_type : str
            Type of the file ('csv' or 'parquet').
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to the respective
            native backend read function, by default None.
        
        Returns
        -------
        Narwhals DataFrame
            The loaded DataFrame wrapped in Narwhals.
        
        Raises
        ------
        ValueError
            If the file_type is not supported.
        
        Examples
        --------
        >>> io = DatasetIO()
        >>> df_csv = io.load_from_file('data.csv', 'csv')
        >>> df_parquet = io.load_from_file('data.parquet', 'parquet')
        """
        if not file_type:
            file_type = Path(file_path).suffix.lstrip('.')

        if file_type.lower() == 'csv':
            return self.load_from_csv(file_path, pl_kwargs)
        elif file_type.lower() == 'parquet':
            return self.load_from_parquet(file_path, pl_kwargs)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}. Supported types are 'csv' and 'parquet'.")

    def save_to_csv(self, df: FrameT, file_path: str, pl_kwargs: dict | None = None ) -> None:
        """Save dataset to a CSV file.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The DataFrame to save.
        file_path : str
            Path where the CSV file will be saved.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to the native DataFrame.write_csv(),
            by default None.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> io = DatasetIO()
        >>> native_df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> df = nw.from_native(native_df)
        >>> io.save_to_csv(df, 'output.csv')
        >>> # Save with custom options
        >>> io.save_to_csv(df, 'output.csv', pl_kwargs={'separator': ';'})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        
        # Convert back to native for I/O operations
        native_df = nw.to_native(df)
        native_df.write_csv(file_path, **(pl_kwargs or {}))
    
    def save_to_parquet(self, df: FrameT, file_path: str, pl_kwargs: dict | None = None ) -> None:
        """Save dataset to a Parquet file.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The DataFrame to save.
        file_path : str
            Path where the Parquet file will be saved.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to the native DataFrame.write_parquet(),
            by default None.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> io = DatasetIO()
        >>> native_df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> df = nw.from_native(native_df)
        >>> io.save_to_parquet(df, 'output.parquet')
        >>> # Save with custom compression
        >>> io.save_to_parquet(df, 'output.parquet', pl_kwargs={'compression': 'snappy'})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert back to native for I/O operations
        native_df = nw.to_native(df)
        native_df.write_parquet(file_path, **(pl_kwargs or {}))