import polars as pl

class DatasetStatistics:
    """Class for computing statistical properties of datasets.
    
    This class provides methods for analyzing correlations, outliers,
    scale variations, and basic descriptive statistics.
    """
    def __init__(self):
        pass
    
    def identify_multicollinearity(self, df, columns, threshold=0.8):
        """Identify pairs of columns with high correlation.
        
        Parameters
        ----------
        df : polars.DataFrame
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
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [2, 4, 6, 8, 10],
        ...     'c': [5, 4, 3, 2, 1]
        ... })
        >>> stats = DatasetStatistics()
        >>> pairs = stats.identify_multicollinearity(df, ['a', 'b', 'c'])
        >>> # Returns pairs where abs(correlation) > 0.8
        """
        df.correlation_matrix = df[columns].corr()
        correlated_pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if abs(df.correlation_matrix.iloc[i, j]) > threshold:
                    correlated_pairs.append((columns[i], columns[j], df.correlation_matrix.iloc[i, j]))

        return correlated_pairs
    
    def varied_scale(self, df, column_name):
        """Calculate the range (scale) of a column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to analyze.
        
        Returns
        -------
        float
            The range of the column (max - min).
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 5, 10, 100]})
        >>> stats = DatasetStatistics()
        >>> scale = stats.varied_scale(df, 'values')
        >>> scale
        99
        """
        col = df.select(pl.col(column_name))
        return col.max()[0, 0] - col.min()[0, 0]
    
    def does_data_have_varied_scale(self, df, column_name, threshold):
        """Check if a column has a scale greater than a threshold.
        
        Parameters
        ----------
        df : polars.DataFrame
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
        >>> import polars as pl
        >>> df = pl.DataFrame({'prices': [10, 20, 1000]})
        >>> stats = DatasetStatistics()
        >>> stats.does_data_have_varied_scale(df, 'prices', 100)
        True
        >>> stats.does_data_have_varied_scale(df, 'prices', 2000)
        False
        """
        scale = self.varied_scale(df, column_name)
        return scale > threshold
    
    def does_data_have_outliers(self, df, column_name, z_threshold=3):
        """Detect outliers using z-score method.
        
        Parameters
        ----------
        df : polars.DataFrame
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
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 3, 4, 100]})
        >>> stats = DatasetStatistics()
        >>> stats.does_data_have_outliers(df, 'values')
        True
        >>> df2 = pl.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats.does_data_have_outliers(df2, 'values')
        False
        """
        col = df.select(pl.col(column_name))
        mean = col.mean()[0, 0]
        std_dev = col.std()[0, 0]
        outliers = col.filter((pl.col(column_name) - mean).abs() > z_threshold * std_dev)
        return outliers.height > 0
    
    def has_outliers(self, df, column_name, z_threshold=3):
        """Detect outliers using z-score method (alias for does_data_have_outliers).
        
        Parameters
        ----------
        df : polars.DataFrame
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
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 3, 4, 100]})
        >>> stats = DatasetStatistics()
        >>> stats.has_outliers(df, 'values')
        True
        """
        return self.does_data_have_outliers(df, column_name, z_threshold)
    
    def get_unq_count(self, df, column_name):
        """Get the count of unique values in a column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to analyze.
        
        Returns
        -------
        int
            The number of unique values in the column.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 2, 3, 3, 3]})
        >>> stats = DatasetStatistics()
        >>> stats.get_unq_count(df, 'values')
        3
        """
        return df.select(pl.col(column_name).n_unique()).item()
    
    def does_data_have_multicollinearity(self, df, columns, threshold=0.8):
        """Check if the dataset has multicollinearity among columns.
        
        Parameters
        ----------
        df : polars.DataFrame
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
        >>> import polars as pl
        >>> df = pl.DataFrame({
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
    
    def take_column_mean(self, df, column_name):
        """Calculate the mean of a column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute mean for.
        
        Returns
        -------
        float
            The mean value of the column.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats = DatasetStatistics()
        >>> stats.take_column_mean(df, 'values')
        3.0
        """
        return df.select(pl.col(column_name).mean()).value()
        
    def take_column_median(self, df, column_name):
        """Calculate the median of a column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute median for.
        
        Returns
        -------
        float
            The median value of the column.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats = DatasetStatistics()
        >>> stats.take_column_median(df, 'values')
        3.0
        """
        return df.select(pl.col(column_name).median()).value()
    
    def take_column_mode(self, df, column_name):
        """Calculate the mode of a column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute mode for.
        
        Returns
        -------
        Any
            The most frequently occurring value in the column.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 2, 3, 3, 3, 4]})
        >>> stats = DatasetStatistics()
        >>> stats.take_column_mode(df, 'values')
        3
        """
        return df.select(pl.col(column_name).mode()).value()
    
    def take_column_std(self, df, column_name):
        """Calculate the standard deviation of a column.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame.
        column_name : str
            Name of the column to compute standard deviation for.
        
        Returns
        -------
        float
            The standard deviation of the column.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> stats = DatasetStatistics()
        >>> std = stats.take_column_std(df, 'values')
        >>> round(std, 2)
        1.58
        """
        return df.select(pl.col(column_name).std()).value()
    
