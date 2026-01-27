import narwhals as nw
from narwhals.typing import FrameT
import polars as pl

class DatasetQuality:
    """
    Class for assessing dataset quality using Narwhals for dataframe abstraction.
    """
    def __init__(self):
        pass

    def get_column_null_count(self, df: FrameT, column: str) -> int:
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
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'feature1': [1, None, 3],
        ...     'feature2': [None, 2, 3]
        ... })
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> quality.get_column_null_count(df, 'feature1')
        1
        """
        return int(df.select(nw.col(column).is_null().sum()).to_numpy()[0, 0])
    
    def type_of_column(self, df: FrameT, column: str) -> str:
        """Get the data type of a column as a simplified string.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column : str
            Name of the column to check.
        
        Returns
        -------
        str
            The simplified data type: 'int', 'float', 'string', or the type name.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'int_col': [1, 2, 3],
        ...     'float_col': [1.0, 2.5, 3.3],
        ...     'str_col': ['a', 'b', 'c']
        ... })
        >>> df = nw.from_native(native_df)
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
        
        # Map types to simplified types
        if 'int' in dtype_str:
            return 'int'
        elif 'float' in dtype_str or 'double' in dtype_str:
            return 'float'
        elif 'str' in dtype_str or 'utf8' in dtype_str or 'string' in dtype_str:
            return 'string'
        else:
            return dtype_str
    
    def has_inconsistent_categorical_data(self, df: FrameT, column: str, threshold: float = 0.1) -> bool:
        """Check if a categorical column has inconsistent data.
        
        Detects potential inconsistencies by checking for categories that differ
        only in case or whitespace, and checks if the number of unique values
        is suspiciously high relative to the dataset size.
        
        Parameters
        ----------
        df : Narwhals DataFrame
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
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'category': ['Cat', 'cat', 'CAT', 'Dog', 'dog']
        ... })
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> quality.has_inconsistent_categorical_data(df, 'category')
        True
        
        >>> native_df2 = pl.DataFrame({
        ...     'category': ['Cat', 'Cat', 'Dog', 'Dog', 'Bird']
        ... })
        >>> df2 = nw.from_native(native_df2)
        >>> quality.has_inconsistent_categorical_data(df2, 'category')
        False
        """
        # Get unique values - convert to native for to_list()
        native_df = nw.to_native(df)
        unique_values = native_df.select(native_df[column].unique()).to_series().to_list()
        
        # Check for case inconsistencies
        normalized = [str(v).strip().lower() if v is not None else None for v in unique_values]
        unique_normalized = set(normalized)
        
        # If normalized count is less than original count, there are case/whitespace inconsistencies
        if len(unique_normalized) < len([v for v in unique_values if v is not None]):
            return True
        
        # Check if the unique ratio is suspiciously high (too many unique values)
        unique_count = int(df.select(nw.col(column).n_unique()).to_numpy()[0, 0])
        total_count = len(df)
        if unique_count / total_count > threshold:
            return True
        
        return False

    def is_text_column(self, df: FrameT, column: str, min_avg_length: int = 50) -> bool:
        """Check if a string column contains text data (e.g., for NLP tasks).
        
        A column is considered a text column if it's a string type and the average
        non-null string length exceeds the minimum threshold.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column : str
            Name of the column to check.
        min_avg_length : int, optional
            Minimum average string length to be considered text. Defaults to 50.
        
        Returns
        -------
        bool
            True if the column appears to contain text data, False otherwise.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'short_text': ['cat', 'dog', 'bird'],
        ...     'long_text': ['This is a long sentence about machine learning.',
        ...                   'Another detailed description of data processing.',
        ...                   'Text classification requires proper preprocessing.']
        ... })
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> quality.is_text_column(df, 'short_text')
        False
        >>> quality.is_text_column(df, 'long_text')
        True
        """
        # Check if column is string type
        dtype_str = str(df.schema[column]).lower()
        if not ('str' in dtype_str or 'utf8' in dtype_str or 'string' in dtype_str):
            return False
        
        # Calculate average string length of non-null values
        native_df = nw.to_native(df)
        col_data = native_df.select(column).to_series()
        
        # Filter out null values and calculate average length
        non_null_values = [str(v) for v in col_data.to_list() if v is not None]
        if not non_null_values:
            return False
        
        avg_length = sum(len(v) for v in non_null_values) / len(non_null_values)
        return avg_length >= min_avg_length

    def check_missing_values(self, df: FrameT) -> FrameT:
        """Check for missing values in the DataFrame.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame to check.
        
        Returns
        -------
        Narwhals DataFrame
            A DataFrame with columns and their corresponding count of missing values.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'feature1': [1, None, 3],
        ...     'feature2': [None, 2, 3]
        ... })
        >>> df = nw.from_native(native_df)
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
            col: int(df.select(nw.col(col).is_null().sum()).to_numpy()[0, 0])
            for col in df.columns
        }
        # Return as native Polars for now (will be wrapped if needed)
        result_df = pl.DataFrame({
            "column": list(missing_counts.keys()),
            "missing_count": list(missing_counts.values())
        })
        return nw.from_native(result_df)
    
    def convert_columns_conservatively_to_best_type(self, df: FrameT) -> FrameT:
        """Convert columns to their best possible types conservatively.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame to convert.
        
        Returns
        -------
        Narwhals DataFrame
            The DataFrame with columns converted to best possible types.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'int_str': ['1', '2', '3'],
        ...     'float_str': ['1.0', '2.5', '3.3'],
        ...     'mixed_str': ['1', 'two', '3']
        ... })
        >>> df = nw.from_native(native_df)
        >>> quality = DatasetQuality()
        >>> converted_df = quality.convert_columns_conservatively_to_best_type(df)
        >>> # Check types
        """
        # This method uses Polars-specific casting, convert to native
        native_df = nw.to_native(df)
        for col in native_df.columns:
            try:
                native_df = native_df.with_columns(native_df[col].cast(pl.Int64, strict=False))
                continue
            except:
                pass
            try:
                native_df = native_df.with_columns(native_df[col].cast(pl.Float64, strict=False))
                continue
            except:
                pass

            try:
                native_df = native_df.with_columns(native_df[col].str.strptime(pl.Datetime, strict=False))
                continue
            except:
                pass

        return nw.from_native(native_df)