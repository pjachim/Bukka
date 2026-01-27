import narwhals as nw
from narwhals.typing import FrameT
from pathlib import Path
from typing import Any

class DatasetIO:
    """Class to handle dataset input/output operations.
    
    This class provides methods for loading and saving datasets in various
    formats including CSV and Parquet using Narwhals for dataframe abstraction.
    """
    def __init__(self):
        pass

    def load_from_csv(self, file_path: str, backend: str | None = None, read_kwargs: dict | None = None) -> FrameT:
        """Load dataset from a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file to load.
        backend : str | None, optional
            Backend name for Narwhals (e.g., 'modin', 'cudf', 'dask', 'pyarrow').
            If None, Narwhals will auto-detect an available backend.
        read_kwargs : dict | None, optional
            Additional keyword arguments to pass to Narwhals `read_csv`,
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
        >>> df = io.load_from_csv('data.csv', read_kwargs={'separator': ';', 'has_header': True})
        """
        if read_kwargs is None:
            read_kwargs = {}
        # Load using Narwhals, optionally selecting a backend via CLI
        return nw.read_csv(file_path, backend=backend, **(read_kwargs or {}))
    
    def load_from_parquet(self, file_path: str, backend: str | None = None, read_kwargs: dict | None = None) -> FrameT:
        """Load dataset from a Parquet file.
        
        Parameters
        ----------
        file_path : str
            Path to the Parquet file to load.
        backend : str | None, optional
            Backend name for Narwhals (e.g., 'modin', 'cudf', 'dask', 'pyarrow').
            If None, Narwhals will auto-detect an available backend.
        read_kwargs : dict | None, optional
            Additional keyword arguments to pass to Narwhals `read_parquet`,
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
        >>> df = io.load_from_parquet('data.parquet', read_kwargs={'columns': ['col1', 'col2']})
        """
        if read_kwargs is None:
            read_kwargs = {}
        # Load using Narwhals, optionally selecting a backend via CLI
        return nw.read_parquet(file_path, backend=backend, **(read_kwargs or {}))
    
    def load_from_file(self, file_path: str, file_type: str | None = None, backend: str | None = None, read_kwargs: dict | None = None) -> FrameT:
        """Load dataset from a file based on its type.
        
        Parameters
        ----------
        file_path : str
            Path to the file to load.
        file_type : str
            Type of the file ('csv' or 'parquet').
        backend : str | None, optional
            Backend name for Narwhals (e.g., 'modin', 'cudf', 'dask', 'pyarrow').
            If None, Narwhals will auto-detect an available backend.
        read_kwargs : dict | None, optional
            Additional keyword arguments to pass to Narwhals read functions,
            by default None.
        
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
            return self.load_from_csv(file_path, backend=backend, read_kwargs=read_kwargs)
        elif file_type.lower() == 'parquet':
            return self.load_from_parquet(file_path, backend=backend, read_kwargs=read_kwargs)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}. Supported types are 'csv' and 'parquet'.")

    def save_to_csv(self, df: FrameT, file_path: str, write_kwargs: dict | None = None ) -> None:
        """Save dataset to a CSV file.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The DataFrame to save.
        file_path : str
            Path where the CSV file will be saved.
        write_kwargs : dict | None, optional
            Additional keyword arguments to pass to the native DataFrame.write_csv(),
            by default None.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import narwhals as nw
        >>> io = DatasetIO()
        >>> df = nw.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> io.save_to_csv(df, 'output.csv')
        >>> # Save with custom options
        >>> io.save_to_csv(df, 'output.csv', write_kwargs={'separator': ';'})
        """
        if write_kwargs is None:
            write_kwargs = {}
        
        # Convert back to native for I/O operations
        native_df = nw.to_native(df)
        native_df.write_csv(file_path, **(write_kwargs or {}))
    
    def save_to_parquet(self, df: FrameT, file_path: str, write_kwargs: dict | None = None ) -> None:
        """Save dataset to a Parquet file.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The DataFrame to save.
        file_path : str
            Path where the Parquet file will be saved.
        write_kwargs : dict | None, optional
            Additional keyword arguments to pass to the native DataFrame.write_parquet(),
            by default None.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import narwhals as nw
        >>> io = DatasetIO()
        >>> df = nw.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> io.save_to_parquet(df, 'output.parquet')
        >>> # Save with custom compression
        >>> io.save_to_parquet(df, 'output.parquet', write_kwargs={'compression': 'snappy'})
        """
        if write_kwargs is None:
            write_kwargs = {}
        
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert back to native for I/O operations
        native_df = nw.to_native(df)
        native_df.write_parquet(file_path, **(write_kwargs or {}))