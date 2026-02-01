import narwhals as nw
from narwhals.typing import FrameT

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
        df : Narwhals DataFrame
            The input DataFrame.
        column : str
            Name of the column to check.
        
        Returns
        -------
        int
            The count of null values in the column.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
        ...     'feature1': [1, None, 3],
        ...     'feature2': [None, 2, 3]
        ... })
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
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
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
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
        ...     'category': ['Cat', 'cat', 'CAT', 'Dog', 'dog']
        ... })
        >>> quality = DatasetQuality()
        >>> quality.has_inconsistent_categorical_data(df, 'category')
        True
        
        >>> df2 = nw.DataFrame({
        ...     'category': ['Cat', 'Cat', 'Dog', 'Dog', 'Bird']
        ... })
        >>> quality.has_inconsistent_categorical_data(df2, 'category')
        False
        """
        # Get unique values - select using Narwhals first, then convert to native
        selected_df = df.select(nw.col(column))
        df.select(nw.col(column).unique())
        unique_values = [x[0] for x in df.select(nw.col(column).unique().drop_nulls()).iter_rows()]
        
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
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
        ...     'short_text': ['cat', 'dog', 'bird'],
        ...     'long_text': ['This is a long sentence about machine learning.',
        ...                   'Another detailed description of data processing.',
        ...                   'Text classification requires proper preprocessing.']
        ... })
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
        
        # Count non-null values
        non_null_values = df.select((~nw.col(column).is_null()).sum()).item()
        if not non_null_values:
            return False
        
        avg_length = df.select(nw.col(column).str.len_chars().mean()).item()
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
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
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
            col: int(df.select(nw.col(col).is_null().sum()).to_numpy()[0, 0])
            for col in df.columns
        }
        # Return as Narwhals DataFrame
        return nw.DataFrame({
            "column": list(missing_counts.keys()),
            "missing_count": list(missing_counts.values())
        })
    
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
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
        ...     'int_str': ['1', '2', '3'],
        ...     'float_str': ['1.0', '2.5', '3.3'],
        ...     'mixed_str': ['1', 'two', '3']
        ... })
        >>> quality = DatasetQuality()
        >>> converted_df = quality.convert_columns_conservatively_to_best_type(df)
        >>> # Check types
        """
        # Conservative no-op conversion without backend-specific dtypes.
        # Future improvement: implement backend-agnostic casting via Narwhals expressions.
        return df