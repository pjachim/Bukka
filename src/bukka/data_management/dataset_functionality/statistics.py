import narwhals as nw
from narwhals.typing import FrameT

class DatasetStatistics:
    """Class for computing statistical properties of datasets.
    
    This class provides methods for analyzing correlations, outliers,
    scale variations, and basic descriptive statistics using Narwhals
    for dataframe abstraction.
    """
    def __init__(self):
        pass
    
    def identify_multicollinearity(self, df: FrameT, columns: list[str], threshold: float = 0.8) -> list[tuple[str, str, float]]:
        """Identify pairs of columns with high correlation.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame to analyze.
        columns : list[str]
            List of column names to check for multicollinearity.
        threshold : float, optional
            Correlation threshold above which columns are considered
            multicollinear, by default 0.8.
        
        Returns
        -------
        list[tuple[str, str, float]]
            List of tuples containing pairs of correlated columns and their
            correlation coefficient: (column1, column2, correlation).
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [2, 4, 6, 8, 10],
        ...     'c': [5, 4, 3, 2, 1]
        ... })
        >>> stats = DatasetStatistics()
        >>> pairs = stats.identify_multicollinearity(df, ['a', 'b', 'c'])
        >>> # Returns pairs where abs(correlation) > 0.8
        """
        # Filter to only numeric columns
        numeric_dtypes = {nw.Int8, nw.Int16, nw.Int32, nw.Int64, nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64, nw.Float32, nw.Float64}
        schema = df.schema
        numeric_columns = [col for col in columns if col in schema and schema[col] in numeric_dtypes]
        
        if len(numeric_columns) < 2:
            # Need at least 2 numeric columns for correlation
            return []
        
        # Compute correlations manually using Narwhals operations
        # Correlation(X, Y) = Cov(X, Y) / (Std(X) * Std(Y))
        correlated_pairs = []
        
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col_i = numeric_columns[i]
                col_j = numeric_columns[j]
                
                # Get means
                mean_i = df.select(nw.col(col_i).mean()).to_numpy()[0, 0]
                mean_j = df.select(nw.col(col_j).mean()).to_numpy()[0, 0]
                
                # Get standard deviations
                std_i = df.select(nw.col(col_i).std()).to_numpy()[0, 0]
                std_j = df.select(nw.col(col_j).std()).to_numpy()[0, 0]
                
                # Skip if either has zero standard deviation
                if std_i == 0 or std_j == 0:
                    continue
                
                # Compute covariance: E[(X - mean_X) * (Y - mean_Y)]
                cov_value = df.select(
                    ((nw.col(col_i) - mean_i) * (nw.col(col_j) - mean_j)).mean()
                ).to_numpy()[0, 0]
                
                # Compute correlation
                corr_value = cov_value / (std_i * std_j)
                
                if abs(corr_value) > threshold:
                    correlated_pairs.append((col_i, col_j, corr_value))
        
        return correlated_pairs
    
    def varied_scale(self, df: FrameT, column_name: str) -> float:
        """Calculate the range (scale) of a column.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to analyze.
        
        Returns
        -------
        float
            The range of the column (max - min).
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 5, 10, 100]})
        >>> stats = DatasetStatistics()
        >>> scale = stats.varied_scale(df, 'values')
        >>> scale
        99
        """
        col = df.select(nw.col(column_name))
        max_val = col.select(nw.col(column_name).max()).to_numpy()[0, 0]
        min_val = col.select(nw.col(column_name).min()).to_numpy()[0, 0]
        return max_val - min_val
    
    def does_data_have_varied_scale(self, df: FrameT, column_name: str, threshold: float) -> bool:
        """Check if a column has a scale greater than a threshold.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to check.
        threshold : float
            The threshold value for determining varied scale.
        
        Returns
        -------
        bool
            True if the column's range exceeds the threshold, False otherwise.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'prices': [10, 20, 1000]})
        >>> stats = DatasetStatistics()
        >>> stats.does_data_have_varied_scale(df, 'prices', 100)
        True
        >>> stats.does_data_have_varied_scale(df, 'prices', 2000)
        False
        """
        scale = self.varied_scale(df, column_name)
        return scale > threshold
    
    def does_data_have_outliers(self, df: FrameT, column_name: str, z_threshold: float = 3) -> bool:
        """Detect outliers using z-score method.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to check for outliers.
        z_threshold : float, optional
            Number of standard deviations from the mean beyond which
            values are considered outliers, by default 3.
        
        Returns
        -------
        bool
            True if outliers are detected, False otherwise.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 3, 4, 100]})
        >>> stats = DatasetStatistics()
        >>> stats.does_data_have_outliers(df, 'values')
        True
        >>> df2 = nw.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats.does_data_have_outliers(df2, 'values')
        False
        """
        col = df.select(nw.col(column_name))
        mean = col.select(nw.col(column_name).mean()).to_numpy()[0, 0]
        std_dev = col.select(nw.col(column_name).std()).to_numpy()[0, 0]
        outliers = col.filter((nw.col(column_name) - mean).abs() > z_threshold * std_dev)
        return len(outliers) > 0
    
    def has_outliers(self, df: FrameT, column_name: str, z_threshold: float = 3) -> bool:
        """Detect outliers using z-score method (alias for does_data_have_outliers).
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to check for outliers.
        z_threshold : float, optional
            Number of standard deviations from the mean beyond which
            values are considered outliers, by default 3.
        
        Returns
        -------
        bool
            True if outliers are detected, False otherwise.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 3, 4, 100]})
        >>> stats = DatasetStatistics()
        >>> stats.has_outliers(df, 'values')
        True
        """
        return self.does_data_have_outliers(df, column_name, z_threshold)
    
    def get_unq_count(self, df: FrameT, column_name: str) -> int:
        """Get the count of unique values in a column.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to analyze.
        
        Returns
        -------
        int
            The number of unique values in the column.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 2, 3, 3, 3]})
        >>> stats = DatasetStatistics()
        >>> stats.get_unq_count(df, 'values')
        3
        """
        return int(df.select(nw.col(column_name).n_unique()).to_numpy()[0, 0])
    
    def does_data_have_multicollinearity(self, df: FrameT, columns: list[str], threshold: float = 0.8) -> bool:
        """Check if the dataset has multicollinearity among columns.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        columns : list[str]
            List of column names to check for multicollinearity.
        threshold : float, optional
            Correlation threshold for detecting multicollinearity,
            by default 0.8.
        
        Returns
        -------
        bool
            True if any pair of columns has correlation above threshold,
            False otherwise.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({
        ...     'a': [1, 2, 3, 4],
        ...     'b': [2, 4, 6, 8],
        ...     'c': [10, 20, 15, 25]
        ... })
        >>> stats = DatasetStatistics()
        >>> stats.does_data_have_multicollinearity(df, ['a', 'b', 'c'])
        True  # 'a' and 'b' are perfectly correlated
        """
        correlated_pairs = self.identify_multicollinearity(df, columns, threshold)
        return len(correlated_pairs) > 0
    
    def take_column_mean(self, df: FrameT, column_name: str) -> float:
        """Calculate the mean of a column.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute mean for.
        
        Returns
        -------
        float
            The mean value of the column.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats = DatasetStatistics()
        >>> stats.take_column_mean(df, 'values')
        3.0
        """
        return float(df.select(nw.col(column_name).mean()).to_numpy()[0, 0])
        
    def take_column_median(self, df: FrameT, column_name: str) -> float:
        """Calculate the median of a column.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute median for.
        
        Returns
        -------
        float
            The median value of the column.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats = DatasetStatistics()
        >>> stats.take_column_median(df, 'values')
        3.0
        """
        return float(df.select(nw.col(column_name).median()).to_numpy()[0, 0])
    
    def take_column_mode(self, df: FrameT, column_name: str):
        """Calculate the mode of a column.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute mode for.
        
        Returns
        -------
        Any
            The most frequently occurring value in the column.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 2, 3, 3, 3, 4]})
        >>> stats = DatasetStatistics()
        >>> stats.take_column_mode(df, 'values')
        3
        """
        # Compute mode using Narwhals: group by value and count, then find max
        # This approach works across all backends
        mode_df = df.select(nw.col(column_name)).group_by(column_name).agg(
            nw.col(column_name).count().alias('count')
        ).sort('count', descending=True).head(1)
        
        # Extract the mode value
        result = mode_df.to_numpy()
        if len(result) > 0:
            return result[0, 0]
        return None
    
    def take_column_std(self, df: FrameT, column_name: str) -> float:
        """Calculate the standard deviation of a column.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute standard deviation for.
        
        Returns
        -------
        float
            The standard deviation of the column.
        
        Examples
        --------
        >>> import narwhals as nw
        >>> df = nw.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats = DatasetStatistics()
        >>> std = stats.take_column_std(df, 'values')
        >>> round(std, 2)
        1.58
        """
        return float(df.select(nw.col(column_name).std()).to_numpy()[0, 0])
    
