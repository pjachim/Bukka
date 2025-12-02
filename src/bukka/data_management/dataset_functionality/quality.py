import polars as pl

class DatasetQuality:
    """
    Class for assessing dataset quality.
    """

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