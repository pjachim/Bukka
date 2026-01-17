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
        
        Returns
        -------
        tuple
            A tuple containing the training and testing DataFrames.
        
        Examples
        --------
        >>> import polars as pl
        >>> import narwhals as nw
        >>> native_df = pl.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [5, 4, 3, 2, 1],
        ...     'target': [0, 1, 0, 1, 0]
        ... })
        >>> df = nw.from_native(native_df)
        >>> manager = DatasetManagement()
        >>> train_df, test_df = manager.split_dataset(df, 'target', train_size=0.6)
        >>> len(train_df)
        3
        >>> len(test_df)
        2
        """
        # Use native backend for shuffling since Narwhals sample may not have fraction param
        shuffled_df = df.sample(fraction=1.0)
        total_len = len(shuffled_df)
        train_len = int(total_len * train_size)
        
        train_df = shuffled_df.head(train_len)
        test_df = shuffled_df.tail(total_len - train_len)

        return train_df, test_df