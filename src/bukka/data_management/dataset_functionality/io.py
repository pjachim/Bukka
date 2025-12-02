import polars as pl

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