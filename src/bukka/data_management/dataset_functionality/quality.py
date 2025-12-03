import polars as pl

class DatasetQuality:
    """
    Class for assessing dataset quality.
    """
    def __init__(self):
        pass

    def get_column_null_count(self, df: pl.DataFrame, column: str) -> int:
        """Get the count of null values in a specific column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column : str
            Name of the column to check.
        
        Returns
        -------
        int
            The count of null values in the column.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'feature1': [1, None, 3],
        ...     'feature2': [None, 2, 3]
        ... })
        >>> quality = DatasetQuality()
        >>> quality.get_column_null_count(df, 'feature1')
        1
        """
        return df.select(pl.col(column).is_null().sum()).item()
    
    def type_of_column(self, df: pl.DataFrame, column: str) -> str:
        """Get the data type of a column as a simplified string.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column : str
            Name of the column to check.
        
        Returns
        -------
        str
            The simplified data type: 'int', 'float', 'string', or the polars type name.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'int_col': [1, 2, 3],
        ...     'float_col': [1.0, 2.5, 3.3],
        ...     'str_col': ['a', 'b', 'c']
        ... })
        >>> quality = DatasetQuality()
        >>> quality.type_of_column(df, 'int_col')
        'int'
        >>> quality.type_of_column(df, 'float_col')
        'float'
        >>> quality.type_of_column(df, 'str_col')
        'string'
        """
        dtype = df.schema[column]
        dtype_str = str(dtype).lower()
        
        # Map polars types to simplified types
        if 'int' in dtype_str:
            return 'int'
        elif 'float' in dtype_str or 'double' in dtype_str:
            return 'float'
        elif 'str' in dtype_str or 'utf8' in dtype_str:
            return 'string'
        else:
            return dtype_str
    
    def has_inconsistent_categorical_data(self, df: pl.DataFrame, column: str, threshold: float = 0.1) -> bool:
        """Check if a categorical column has inconsistent data.
        
        Detects potential inconsistencies by checking for categories that differ
        only in case or whitespace, and checks if the number of unique values
        is suspiciously high relative to the dataset size.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column : str
            Name of the column to check.
        threshold : float, optional
            Threshold for unique value ratio (unique_values / total_rows).
            If ratio exceeds this, data may be inconsistent. Defaults to 0.1.
        
        Returns
        -------
        bool
            True if inconsistent categorical data is detected, False otherwise.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'category': ['Cat', 'cat', 'CAT', 'Dog', 'dog']
        ... })
        >>> quality = DatasetQuality()
        >>> quality.has_inconsistent_categorical_data(df, 'category')
        True
        
        >>> df2 = pl.DataFrame({
        ...     'category': ['Cat', 'Cat', 'Dog', 'Dog', 'Bird']
        ... })
        >>> quality.has_inconsistent_categorical_data(df2, 'category')
        False
        """
        # Get unique values
        unique_values = df.select(pl.col(column).unique()).to_series().to_list()
        
        # Check for case inconsistencies
        normalized = [str(v).strip().lower() if v is not None else None for v in unique_values]
        unique_normalized = set(normalized)
        
        # If normalized count is less than original count, there are case/whitespace inconsistencies
        if len(unique_normalized) < len([v for v in unique_values if v is not None]):
            return True
        
        # Check if the unique ratio is suspiciously high (too many unique values)
        unique_count = df.select(pl.col(column).n_unique()).item()
        total_count = df.height
        if unique_count / total_count > threshold:
            return True
        
        return False

    def check_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Check for missing values in the DataFrame.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame to check.
        
        Returns
        -------
        polars.DataFrame
            A DataFrame with columns and their corresponding count of missing values.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'feature1': [1, None, 3],
        ...     'feature2': [None, 2, 3]
        ... })
        >>> quality = DatasetQuality()
        >>> missing_df = quality.check_missing_values(df)
        >>> missing_df
        shape: (2, 2)
        ┌────────────┬───────────────┐
        │ column     ┆ missing_count │
        │ ---        ┆ ---           │
        │ str        ┆ u32           │
        ╞════════════╪═══════════════╡
        │ feature1   ┆ 1             │
        │ feature2   ┆ 1             │
        └────────────┴───────────────┘
        """
        missing_counts = {
            col: df.select(pl.col(col).is_null().sum()).item()
            for col in df.columns
        }
        return pl.DataFrame({
            "column": list(missing_counts.keys()),
            "missing_count": list(missing_counts.values())
        })
    
    def convert_columns_conservatively_to_best_type(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert columns to their best possible types conservatively.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame to convert.
        
        Returns
        -------
        polars.DataFrame
            The DataFrame with columns converted to best possible types.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'int_str': ['1', '2', '3'],
        ...     'float_str': ['1.0', '2.5', '3.3'],
        ...     'mixed_str': ['1', 'two', '3']
        ... })
        >>> quality = DatasetQuality()
        >>> converted_df = quality.convert_columns_conservatively_to_best_type(df)
        >>> converted_df.dtypes
        [Int64, Float64, Utf8]
        """
        for col in df.columns:
            try:
                df = df.with_column(pl.col(col).cast(pl.Int64, strict=False))
                continue
            except:
                pass
            try:
                df = df.with_column(pl.col(col).cast(pl.Float64, strict=False))
                continue
            except:
                pass

            try:
                df = df.with_column(pl.col(col).str.strptime(pl.Datetime, strict=False))
                continue
            except:
                pass

        return df