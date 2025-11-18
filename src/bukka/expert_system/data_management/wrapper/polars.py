import polars as pl
from pathlib import Path
import math

class PolarsOperations:
    def __init__(self, train_df=None, full_df:pl.DataFrame | None=None, random_seed: int | None=None):
        self.train_df = train_df
        self.full_df = full_df
        self.random_seed = random_seed

    def read_csv(self, path):
        self.train_df = pl.read_csv(path)

    def read_parquet(self, path):
        self.train_df = pl.read_parquet(path)

    def split_dataset(self, train_path, test_path, target_column: str, train_size: float=0.8, stratify=True, strata: list[str] | None = None):
        self.full_df = self.full_df.sample(shuffle=True, fraction=1.0)

        if stratify:
            train_results: list[pl.DataFrame] = []
            test_results: list[pl.DataFrame] = []

            if strata is None:
                strata = [target_column]

            for df in self.full_df.partition_by(strata):
                train, test = self._split_dataset(df, train_size=train_size)
                train_results.append(train)
                test_results.append(test)

            train_df = pl.concat(train_results).full_df.sample(shuffle=True, fraction=1.0)
            test_df = pl.concat(test_results).full_df.sample(shuffle=True, fraction=1.0)

        else:
            train_df, test_df = self._split_dataset(self.full_df, train_size=train_size)

        self.write_parquet(train_df, train_path)
        self.write_parquet(test_df, test_path)

    def write_parquet(self, df: pl.DataFrame, path):
        df.write_parquet(path)

    def _split_dataset(self, df:pl.DataFrame, train_size: float) -> tuple[pl.DataFrame, pl.DataFrame]:
        n_train = math.ceil(df.height * train_size)
        n_test = df.height - n_train
        return df.slice(0, n_train), df.slice(n_train, n_test)

    def get_column_names(self):
        return self.train_df.columns
    
    def get_dtypes(self):
        return dict(self.schema)

    def get_column_mean(self, column):
        self._ensure_df()
        return self.train_df.select(pl.col(column).mean()).item()

    def get_column_std(self, column):
        self._ensure_df()
        return self.train_df.select(pl.col(column).std()).item()

    def get_column_median(self, column):
        self._ensure_df()
        return self.train_df.select(pl.col(column).median()).item()

    def get_column_null_count(self, column):
        self._ensure_df()
        return self.train_df.select(pl.col(column).is_null().sum()).item()

    def get_number_of_samples(self):
        return self.train_df.height
    
    def _ensure_df(self):
        if self.train_df is None:
            raise AttributeError('No self.df attribute. Call a method with the prefix read_, and try again.')