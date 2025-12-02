import polars as pl
from pathlib import Path

class DatasetIO:
    """Class to handle dataset input/output operations.
    
    This class provides methods for loading and saving datasets in various
    formats including CSV and Parquet.
    """
    def load_from_csv(self, file_path: str, pl_kwargs: dict | None = None) -> pl.DataFrame:
        """Load dataset from a CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file to load.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to polars.read_csv(),
            by default None.
        
        Returns
        -------
        polars.DataFrame
            The loaded DataFrame.
        
        Examples
        --------
        >>> io = DatasetIO()
        >>> df = io.load_from_csv('data.csv')
        >>> # Load with custom options
        >>> df = io.load_from_csv('data.csv', pl_kwargs={'separator': ';', 'has_header': True})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        return pl.read_csv(file_path, **(pl_kwargs or {}))
    
    def load_from_parquet(self, file_path: str, pl_kwargs: dict | None = None) -> pl.DataFrame:
        """Load dataset from a Parquet file.
        
        Parameters
        ----------
        file_path : str
            Path to the Parquet file to load.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to polars.read_parquet(),
            by default None.
        
        Returns
        -------
        polars.DataFrame
            The loaded DataFrame.
        
        Examples
        --------
        >>> io = DatasetIO()
        >>> df = io.load_from_parquet('data.parquet')
        >>> # Load with custom options
        >>> df = io.load_from_parquet('data.parquet', pl_kwargs={'columns': ['col1', 'col2']})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        return pl.read_parquet(file_path, **(pl_kwargs or {}))
    
    def load_from_file(self, file_path: str, file_type: str | None = None, pl_kwargs: dict | None = None) -> pl.DataFrame:
        """Load dataset from a file based on its type.
        
        Parameters
        ----------
        file_path : str
            Path to the file to load.
        file_type : str
            Type of the file ('csv' or 'parquet').
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to the respective
            Polars read function, by default None.
        
        Returns
        -------
        polars.DataFrame
            The loaded DataFrame.
        
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

    def save_to_csv(self, df: pl.DataFrame, file_path: str, pl_kwargs: dict | None = None ) -> None:
        """Save dataset to a CSV file.
        
        Parameters
        ----------
        df : polars.DataFrame
            The DataFrame to save.
        file_path : str
            Path where the CSV file will be saved.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to DataFrame.write_csv(),
            by default None.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import polars as pl
        >>> io = DatasetIO()
        >>> df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> io.save_to_csv(df, 'output.csv')
        >>> # Save with custom options
        >>> io.save_to_csv(df, 'output.csv', pl_kwargs={'separator': ';'})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        
        df.write_csv(file_path, **(pl_kwargs or {}))
    
    def save_to_parquet(self, df: pl.DataFrame, file_path: str, pl_kwargs: dict | None = None ) -> None:
        """Save dataset to a Parquet file.
        
        Parameters
        ----------
        df : polars.DataFrame
            The DataFrame to save.
        file_path : str
            Path where the Parquet file will be saved.
        pl_kwargs : dict | None, optional
            Additional keyword arguments to pass to DataFrame.write_parquet(),
            by default None.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> import polars as pl
        >>> io = DatasetIO()
        >>> df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> io.save_to_parquet(df, 'output.parquet')
        >>> # Save with custom compression
        >>> io.save_to_parquet(df, 'output.parquet', pl_kwargs={'compression': 'snappy'})
        """
        if pl_kwargs is None:
            pl_kwargs = {}
        
        df.write_parquet(file_path, **(pl_kwargs or {}))