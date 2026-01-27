import narwhals as nw
from narwhals.typing import FrameT

class DatasetManagement:
    """
    Class for managing dataset functionalities using Narwhals for dataframe abstraction.
    """
    def __init__(self):
        pass

    def split_dataset(self, df: FrameT, target_column: str, train_size: float = 0.8, strata: list[str] | None = None, stratify: bool = True, target_dataframe: FrameT | None = None) -> tuple[FrameT, FrameT]:
        """Split the dataset into training and testing sets.
        
        Parameters
        ----------
        df : Narwhals DataFrame
            The input DataFrame to split.
        target_column : str
            The name of the target column.
        train_size : float, optional
            Proportion of the dataset to include in the training set,
            by default 0.8.
        strata : list[str] | None, optional
            Columns to use for stratification. If None and stratify is True,
            uses the target_column.
        stratify : bool, optional
            Whether to perform stratified sampling, by default True.
        target_dataframe : FrameT | None, optional
            Not used in current implementation.
        
        Returns
        -------
        tuple
            A tuple containing the training and testing DataFrames.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        ...     'feature2': [5, 4, 3, 2, 1, 8, 7, 6],
        ...     'target': [0, 1, 0, 1, 0, 1, 0, 1]
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'target', train_size=0.5, stratify=True)
        >>> # Check stratification: both sets should have equal proportions of each class
        >>> train_df.group_by('target').len().sort('target')  # doctest: +SKIP
        >>> test_df.group_by('target').len().sort('target')  # doctest: +SKIP
        """
        if not stratify:
            # Simple random split without stratification
            shuffled_df = df.sample(fraction=1.0)
            total_len = len(shuffled_df)
            train_len = int(total_len * train_size)
            
            train_df = shuffled_df.head(train_len)
            test_df = shuffled_df.tail(total_len - train_len)
            return train_df, test_df
        
        # Stratified split
        stratify_cols = strata if strata is not None else [target_column]
        
        # Check if stratification makes sense (groups should have multiple samples)
        unique_count = int(df.select(nw.col(stratify_cols[0]).n_unique()).to_numpy()[0, 0])
        total_rows = len(df)
        
        # If almost all values are unique, stratification doesn't make sense
        # Fall back to simple random split
        if unique_count / total_rows > 0.5:
            shuffled_df = df.sample(fraction=1.0)
            total_len = len(shuffled_df)
            train_len = int(total_len * train_size)
            
            train_df = shuffled_df.head(train_len)
            test_df = shuffled_df.tail(total_len - train_len)
            return train_df, test_df
        
        # Group by stratification columns
        groups = df.group_by(stratify_cols)
        
        train_parts = []
        test_parts = []
        
        for group_key, group_df in groups:
            # Shuffle each group
            shuffled_group = group_df.sample(fraction=1.0)
            group_len = len(shuffled_group)
            group_train_len = int(group_len * train_size)
            
            # Ensure at least something goes into each split if group is large enough
            if group_len >= 2:
                group_train_len = max(1, group_train_len)  # At least 1 in train if group >= 2
                # Split this group
                train_parts.append(shuffled_group.head(group_train_len))
                test_parts.append(shuffled_group.tail(group_len - group_train_len))
            elif group_len == 1:
                # Single-row groups go to train with probability = train_size
                import random
                if random.random() < train_size:
                    train_parts.append(shuffled_group)
                else:
                    test_parts.append(shuffled_group)
        
        # Concatenate all groups (handle empty lists)
        train_df = nw.concat(train_parts) if train_parts else df.head(0)
        test_df = nw.concat(test_parts) if test_parts else df.head(0)
        
        # Final shuffle to avoid having groups clustered together
        train_df = train_df.sample(fraction=1.0)
        test_df = test_df.sample(fraction=1.0)

        return train_df, test_df