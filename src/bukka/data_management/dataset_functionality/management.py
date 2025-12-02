import polars as pl

class DatasetManagement:
    """
    Class for managing dataset functionalities.
    """

    def split_dataset(self, df: pl.DataFrame, target_column: str, train_size=0.8, strata: list[str] | None = None, stratify: bool = True, target_dataframe: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame] :
        """Split the dataset into training and testing sets.
        
        Parameters
        ----------
        df : polars.DataFrame
            The input DataFrame to split.
        target_column : str
            The name of the target column.
        train_size : float, optional
            Proportion of the dataset to include in the training set,
            by default 0.8.
        
        Returns
        -------
        tuple
            A tuple containing the training and testing DataFrames.
        
        Examples
        --------
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [5, 4, 3, 2, 1],
        ...     'target': [0, 1, 0, 1, 0]
        ... })
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'target', train_size=0.6)
        >>> len(train_df)
        3
        >>> len(test_df)
        2
        """
        shuffled_df = df.sample(frac=1.0, with_replacement=False)
        train_df = shuffled_df.head(int(len(shuffled_df) * train_size))
        test_df = shuffled_df.tail(len(shuffled_df) - len(train_df))
        
        return train_df, test_df