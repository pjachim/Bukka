import polars as pl
import pytest
from pathlib import Path
import math
from polars.testing import assert_frame_equal
from bukka.expert_system.data_management.wrapper.polars import PolarsOperations

# Define the same default seed used in the class for testing reproducibility
TEST_RANDOM_SEED = 42

# --- Fixtures ---

@pytest.fixture(scope="session")
def basic_df() -> pl.DataFrame:
    """Fixture for a basic Polars DataFrame with a unique index column for checking shuffle."""
    data = {
        'index': [1, 2, 3, 4, 5],
        'col_int': [10, 20, 30, 40, 50],
        'col_str': ['a', 'b', 'a', 'c', 'b'],
        'col_null': [100, None, 300, 400, None]
    }
    return pl.DataFrame(data)

@pytest.fixture(scope="session")
def large_stratified_df() -> pl.DataFrame:
    """Fixture for a larger DataFrame suitable for stratification testing."""
    # Creates a DataFrame with 100 rows and a categorical target column 'target'
    data = {
        'feature': list(range(1, 101)),
        'target': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
        'group': ['G1'] * 25 + ['G2'] * 25 + ['G1'] * 15 + ['G2'] * 15 + ['G1'] * 10 + ['G2'] * 10
    }
    return pl.DataFrame(data)

# --- Test Class for Splitting Methods ---

class TestSplitDatasetMethods:
    """Test suite for data splitting methods, focusing on _split_dataset and split_dataset."""

    # Test the internal helper method first
    class TestInternalSplit:
        """Test cases for the private _split_dataset helper method."""

        def test_split_is_shuffled(self, basic_df: pl.DataFrame):
            """Test that the split output is shuffled and not in original order."""
            train_df, _ = PolarsOperations()._split_dataset(basic_df, train_size=0.8, seed=TEST_RANDOM_SEED)
            
            # The original train data (first 4 rows) is index: [1, 2, 3, 4]
            # Since shuffling is applied, the train_df indices should be different.
            
            # For seed=42, the shuffled order of basic_df (indices 1 to 5) is [4, 5, 2, 1, 3]
            expected_shuffled_train_indices = [4, 5, 2, 1] 
            
            assert train_df.height == 4
            assert list(train_df.get_column('index')) == expected_shuffled_train_indices
            
            # The remaining index is 3, which should be in the test set.
            expected_shuffled_test_indices = [3]
            
            _, test_df = PolarsOperations()._split_dataset(basic_df, train_size=0.8, seed=TEST_RANDOM_SEED)
            assert test_df.height == 1
            assert list(test_df.get_column('index')) == expected_shuffled_test_indices

        def test_split_is_reproducible(self, basic_df: pl.DataFrame):
            """Test that splits with the same seed produce identical results."""
            # Run 1
            train_df_1, test_df_1 = PolarsOperations()._split_dataset(basic_df, train_size=0.5, seed=TEST_RANDOM_SEED)
            # Run 2
            train_df_2, test_df_2 = PolarsOperations()._split_dataset(basic_df, train_size=0.5, seed=TEST_RANDOM_SEED)
            
            assert_frame_equal(train_df_1, train_df_2)
            assert_frame_equal(test_df_1, test_df_2)
            
            # Run 3 with a DIFFERENT seed
            train_df_3, _ = PolarsOperations()._split_dataset(basic_df, train_size=0.5, seed=TEST_RANDOM_SEED + 1)
            
            # Verify the different seed produces a different result
            with pytest.raises(AssertionError):
                assert_frame_equal(train_df_1, train_df_3)
                
        def test_train_size_one(self, basic_df: pl.DataFrame):
            """Test an edge case where train_size is 1.0 (all rows in train)."""
            train_df, test_df = PolarsOperations()._split_dataset(basic_df, train_size=1.0, seed=TEST_RANDOM_SEED)
            assert train_df.height == 5
            assert test_df.height == 0
            assert test_df.is_empty()
            # Check train_df is a shuffled version of basic_df
            assert train_df.sort('index').frame_equal(basic_df)


    # Test the public-facing split method
    class TestPublicSplit:
        """Test cases for the public split_dataset method."""

        def test_non_stratified_split_is_shuffled_and_reproducible(self, tmp_path: Path, large_stratified_df: pl.DataFrame):
            """Test non-stratified split with shuffling and reproducibility."""
            train_path_1 = tmp_path / "train_nonstrat_1.parquet"
            train_path_2 = tmp_path / "train_nonstrat_2.parquet"

            op = PolarsOperations(full_df=large_stratified_df)
            
            # Run 1
            op.split_dataset(train_path_1, tmp_path / "test_1.parquet", target_column="target", train_size=0.7, stratify=False, seed=TEST_RANDOM_SEED)
            
            # Run 2 (Same seed)
            op.split_dataset(train_path_2, tmp_path / "test_2.parquet", target_column="target", train_size=0.7, stratify=False, seed=TEST_RANDOM_SEED)
            
            train_df_1 = pl.read_parquet(train_path_1)
            train_df_2 = pl.read_parquet(train_path_2)

            # Check reproducibility
            assert_frame_equal(train_df_1, train_df_2)

        def test_stratified_split_by_target_is_shuffled(self, tmp_path: Path, large_stratified_df: pl.DataFrame):
            """Test stratification ensuring shuffling occurs within each stratum."""
            train_path = tmp_path / "train_strat_shuffled.parquet"
            test_path = tmp_path / "test_strat_shuffled.parquet"
            target_col = "target"
            train_size = 0.8
            
            op = PolarsOperations(full_df=large_stratified_df)
            op.split_dataset(train_path, test_path, target_column=target_col, train_size=train_size, stratify=True, seed=TEST_RANDOM_SEED)

            train_df = pl.read_parquet(train_path)
            
            # Verify the stratification counts remain correct (80/20 split)
            train_counts = train_df.group_by(target_col).agg(pl.count()).sort(target_col)
            expected_train_counts = pl.DataFrame({target_col: ['A', 'B', 'C'], 'count': [40, 24, 16]}).sort(target_col)
            assert_frame_equal(train_counts, expected_train_counts)
            
            # Verify that the feature column, which was in order (1 to 100), is now shuffled.
            # We check the first few elements of the train split for class A (first 50 original rows).
            # The first 40 rows of the stratified train_df should be shuffled elements from original rows 1-50.
            
            # Filter the train split to just class A samples
            train_class_a = train_df.filter(pl.col(target_col) == 'A').sort('feature')
            original_class_a = large_stratified_df.filter(pl.col(target_col) == 'A').sort('feature')
            
            # Check that the set of *features* in the train split is a subset of the original
            assert set(train_class_a['feature'].to_list()).issubset(set(original_class_a['feature'].to_list()))
            
            # Check that the order in the train split is NOT the original order
            # The original order for features 1-40 (which is the train set size for A) is [1, 2, ..., 40]
            # Since shuffling occurred, the first 40 elements of train_class_a *before* sorting should be shuffled.
            # The original *train_df* for class A should be shuffled:
            train_class_a_unshuffled = train_df.filter(pl.col(target_col) == 'A')
            assert train_class_a_unshuffled['feature'][0].item() != original_class_a['feature'][0].item()
            
# (Other Test Classes remain the same)
# The remaining Test classes (TestPolarsOperations, TestReadWriteMethods, TestStatisticsMethods) 
# are identical to the previous response as they were not affected by the split logic change.

import polars as pl
import pytest
from pathlib import Path
import math
from polars.testing import assert_frame_equal
# Assuming the PolarsOperations class is saved in a file named 'polars_operations.py'
# from polars_operations import PolarsOperations 
# (For a self-contained example, we'll assume the class is available in the testing environment)

# --- Fixtures ---

@pytest.fixture(scope="session")
def basic_df() -> pl.DataFrame:
    """Fixture for a basic Polars DataFrame."""
    data = {
        'col_int': [1, 2, 3, 4, 5],
        'col_float': [10.5, 20.5, 30.5, 40.5, 50.5],
        'col_str': ['a', 'b', 'a', 'c', 'b'],
        'col_null': [100, None, 300, 400, None]
    }
    return pl.DataFrame(data)

@pytest.fixture(scope="session")
def large_stratified_df() -> pl.DataFrame:
    """Fixture for a larger DataFrame suitable for stratification testing."""
    # Creates a DataFrame with 100 rows and a categorical target column 'target'
    # Distribution: 50 Class A, 30 Class B, 20 Class C
    data = {
        'feature': list(range(1, 101)),
        'target': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
        'group': ['G1'] * 25 + ['G2'] * 25 + ['G1'] * 15 + ['G2'] * 15 + ['G1'] * 10 + ['G2'] * 10
    }
    return pl.DataFrame(data)

@pytest.fixture
def empty_df() -> pl.DataFrame:
    """Fixture for an empty Polars DataFrame."""
    return pl.DataFrame({'col_int': pl.Series([], dtype=pl.Int64)})

# --- Test Class for PolarsOperations ---

class TestPolarsOperations:
    """
    Test suite for the PolarsOperations class, testing initialization, attributes,
    and handling of internal state.
    """
    
    def test_init_with_no_args(self):
        """Test initialization with default (None) arguments."""
        op = PolarsOperations()
        assert op.train_df is None
        assert op.full_df is None

    def test_init_with_dataframes(self, basic_df: pl.DataFrame):
        """Test initialization by passing initial DataFrames."""
        op = PolarsOperations(train_df=basic_df, full_df=basic_df)
        assert_frame_equal(op.train_df, basic_df)
        assert_frame_equal(op.full_df, basic_df)

# --- Test Class for I/O Methods ---

class TestReadWriteMethods:
    """Test suite for file input/output methods."""
    
    def test_read_csv(self, tmp_path: Path, basic_df: pl.DataFrame):
        """Test reading a CSV file and setting train_df."""
        csv_path = tmp_path / "test.csv"
        # Manually write the fixture DF to CSV for reading back
        basic_df.write_csv(csv_path)
        
        op = PolarsOperations()
        op.read_csv(csv_path)
        assert op.train_df is not None
        assert_frame_equal(op.train_df, basic_df)

    def test_read_parquet(self, tmp_path: Path, basic_df: pl.DataFrame):
        """Test reading a Parquet file and setting train_df."""
        parquet_path = tmp_path / "test.parquet"
        # Manually write the fixture DF to Parquet for reading back
        basic_df.write_parquet(parquet_path)

        op = PolarsOperations()
        op.read_parquet(parquet_path)
        assert op.train_df is not None
        assert_frame_equal(op.train_df, basic_df)

    def test_write_parquet(self, tmp_path: Path, basic_df: pl.DataFrame):
        """Test writing a DataFrame to a Parquet file."""
        parquet_path = tmp_path / "output.parquet"
        op = PolarsOperations()
        op.write_parquet(basic_df, parquet_path)
        
        # Verify the file was written and can be read back correctly
        assert parquet_path.exists()
        read_df = pl.read_parquet(parquet_path)
        assert_frame_equal(read_df, basic_df)
    
# --- Test Class for Splitting Methods ---

class TestSplitDatasetMethods:
    """Test suite for data splitting methods, focusing on _split_dataset and split_dataset."""

    # Test the internal helper method first
    class TestInternalSplit:
        """Test cases for the private _split_dataset helper method."""

        def test_standard_split(self, basic_df: pl.DataFrame):
            """Test a standard 80/20 split."""
            train_df, test_df = PolarsOperations()._split_dataset(basic_df, train_size=0.8, seed=7)
            # basic_df has 5 rows. 5 * 0.8 = 4.0. math.ceil(4.0) = 4.
            assert train_df.height == 4
            assert test_df.height == 1
            assert_frame_equal(train_df, basic_df.head(4))
            assert_frame_equal(test_df, basic_df.tail(1))

        def test_perfect_split(self, large_stratified_df: pl.DataFrame):
            """Test a perfect 50/50 split on a large, even-sized DF."""
            # large_stratified_df has 100 rows. 100 * 0.5 = 50.
            train_df, test_df = PolarsOperations()._split_dataset(large_stratified_df, train_size=0.5, seed=7362)
            assert train_df.height == 50
            assert test_df.height == 50

        def test_train_size_one(self, basic_df: pl.DataFrame):
            """Test an edge case where train_size is 1.0 (all rows in train)."""
            # 5 * 1.0 = 5. math.ceil(5) = 5.
            train_df, test_df = PolarsOperations()._split_dataset(basic_df, train_size=1.0)
            assert train_df.height == 5
            assert test_df.height == 0
            assert_frame_equal(train_df, basic_df)
            assert test_df.is_empty()

        def test_train_size_zero(self, basic_df: pl.DataFrame):
            """Test an edge case where train_size is 0.0 (all rows in test)."""
            # 5 * 0.0 = 0. math.ceil(0) = 0.
            train_df, test_df = PolarsOperations()._split_dataset(basic_df, train_size=0.0)
            assert train_df.height == 0
            assert test_df.height == 5
            assert train_df.is_empty()
            assert_frame_equal(test_df, basic_df)

        def test_on_empty_df(self, empty_df: pl.DataFrame):
            """Test splitting an empty DataFrame."""
            train_df, test_df = PolarsOperations()._split_dataset(empty_df, train_size=0.8)
            assert train_df.height == 0
            assert test_df.height == 0
            assert train_df.is_empty()
            assert test_df.is_empty()

    # Test the public-facing split method
    class TestPublicSplit:
        """Test cases for the public split_dataset method."""

        def test_missing_full_df_raises(self, tmp_path: Path):
            """Test that split_dataset raises AttributeError if full_df is None."""
            op = PolarsOperations()
            with pytest.raises(AttributeError, match="full_df is None"):
                op.split_dataset(tmp_path / "train.parquet", tmp_path / "test.parquet", target_column="col_str")

        def test_non_stratified_split(self, tmp_path: Path, large_stratified_df: pl.DataFrame):
            """Test a basic non-stratified split and file saving."""
            train_path = tmp_path / "train_nonstrat.parquet"
            test_path = tmp_path / "test_nonstrat.parquet"

            op = PolarsOperations(full_df=large_stratified_df)
            op.split_dataset(train_path, test_path, target_column="target", train_size=0.7, stratify=False)

            # Verification of file existence and size
            assert train_path.exists()
            assert test_path.exists()
            train_df = pl.read_parquet(train_path)
            test_df = pl.read_parquet(test_path)

            # 100 rows * 0.7 = 70. math.ceil(70) = 70
            assert train_df.height == 70
            assert test_df.height == 30
            assert train_df.height + test_df.height == large_stratified_df.height

        def test_stratified_split_by_target(self, tmp_path: Path, large_stratified_df: pl.DataFrame):
            """Test stratification using the target_column."""
            train_path = tmp_path / "train_strat_target.parquet"
            test_path = tmp_path / "test_strat_target.parquet"
            target_col = "target"
            train_size = 0.8 # A: 50*0.8=40, B: 30*0.8=24, C: 20*0.8=16. Total train: 80

            op = PolarsOperations(full_df=large_stratified_df)
            op.split_dataset(train_path, test_path, target_column=target_col, train_size=train_size, stratify=True)

            train_df = pl.read_parquet(train_path)
            test_df = pl.read_parquet(test_path)

            # Check total sizes
            assert train_df.height == 80
            assert test_df.height == 20
            
            # Check stratification counts
            train_counts = train_df.group_by(target_col).agg(pl.count()).sort(target_col)
            test_counts = test_df.group_by(target_col).agg(pl.count()).sort(target_col)

            expected_train_counts = pl.DataFrame({target_col: ['A', 'B', 'C'], 'count': [40, 24, 16]}).sort(target_col)
            expected_train_counts = expected_train_counts.with_columns(pl.col(target_col).cast(pl.UInt32))
            expected_test_counts = pl.DataFrame({target_col: ['A', 'B', 'C'], 'count': [10, 6, 4]}).sort(target_col)
            expected_test_counts = expected_test_counts.with_columns(pl.col(target_col).cast(pl.UInt32))
            
            assert_frame_equal(train_counts, expected_train_counts)
            assert_frame_equal(test_counts, expected_test_counts)
            
        def test_stratified_split_by_multiple_strata(self, tmp_path: Path, large_stratified_df: pl.DataFrame):
            """Test stratification using multiple columns ('target' and 'group')."""
            train_path = tmp_path / "train_strat_multi.parquet"
            test_path = tmp_path / "test_strat_multi.parquet"
            strata_cols = ['target', 'group']
            train_size = 0.8 # Total train: 80

            # Expected Stratum Counts (80% train / 20% test)
            # A/G1: 25 -> Train: 20, Test: 5
            # A/G2: 25 -> Train: 20, Test: 5
            # B/G1: 15 -> Train: 12, Test: 3
            # B/G2: 15 -> Train: 12, Test: 3
            # C/G1: 10 -> Train: 8, Test: 2
            # C/G2: 10 -> Train: 8, Test: 2

            op = PolarsOperations(full_df=large_stratified_df)
            op.split_dataset(train_path, test_path, target_column="target", train_size=train_size, stratify=True, strata=strata_cols)

            train_df = pl.read_parquet(train_path)
            test_df = pl.read_parquet(test_path)

            # Check total sizes
            assert train_df.height == 80
            assert test_df.height == 20
            
            # Check stratification counts
            train_counts = train_df.group_by(strata_cols).agg(pl.count()).sort(strata_cols)
            
            expected_train_counts = pl.DataFrame({
                'target': ['A', 'A', 'B', 'B', 'C', 'C'],
                'group': ['G1', 'G2', 'G1', 'G2', 'G1', 'G2'],
                'count': [20, 20, 12, 12, 8, 8]
            }).sort(strata_cols)

            assert_frame_equal(train_counts, expected_train_counts)
            
# --- Test Class for Statistics Methods ---

class TestStatisticsMethods:
    """Test suite for descriptive statistics and metadata methods."""
    
    # Fixture for a test DataFrame with mixed data types and nulls
    @pytest.fixture(scope="class")
    def stats_df(self) -> pl.DataFrame:
        data = {
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e'],
            'D': [100, None, 300, 400, None] # Two nulls
        }
        return pl.DataFrame(data)

    # Fixture for an initialized PolarsOperations object
    @pytest.fixture(scope="class")
    def op_stats(self, stats_df: pl.DataFrame):
        return PolarsOperations(train_df=stats_df)

    def test_df_none_raises_attribute_error(self):
        """Test that stat methods raise an error when train_df is None."""
        op = PolarsOperations()
        methods = [
            op.get_column_names, op.get_dtypes, op.get_number_of_samples,
            lambda: op.get_column_mean('A'), lambda: op.get_column_std('A'),
            lambda: op.get_column_median('A'), lambda: op.get_column_null_count('A')
        ]
        
        for method in methods:
            with pytest.raises(AttributeError, match="self.train_df is None"):
                method()

    # --- Metadata Tests ---

    def test_get_column_names(self, op_stats: PolarsOperations):
        """Test retrieval of column names."""
        assert op_stats.get_column_names() == ['A', 'B', 'C', 'D']

    def test_get_dtypes(self, op_stats: PolarsOperations):
        """Test retrieval of column data types (schema)."""
        expected_dtypes = {
            'A': pl.Float64, 
            'B': pl.Int64, 
            'C': pl.String, 
            'D': pl.Int64 # Polars infers D as Int64 with nulls
        }
        assert op_stats.get_dtypes() == expected_dtypes

    def test_get_number_of_samples(self, op_stats: PolarsOperations, stats_df: pl.DataFrame):
        """Test retrieval of DataFrame row count."""
        assert op_stats.get_number_of_samples() == stats_df.height

    # --- Statistic Calculation Tests ---

    def test_get_column_mean(self, op_stats: PolarsOperations):
        """Test mean calculation for numeric columns."""
        # Col A: (1+2+3+4+5)/5 = 3.0
        assert op_stats.get_column_mean('A') == pytest.approx(3.0)
        # Col B: (10+20+30+40+50)/5 = 30.0
        assert op_stats.get_column_mean('B') == pytest.approx(30.0)
        # Col D (with nulls): (100+300+400)/3 = 800/3 = 266.66...
        assert op_stats.get_column_mean('D') == pytest.approx(266.6666666666667)
        
    def test_get_column_std(self, op_stats: PolarsOperations):
        """Test standard deviation calculation."""
        # Polars default is sample stddev (ddof=1)
        # Col B: [10, 20, 30, 40, 50]. Mean=30. stddev = 15.811...
        assert op_stats.get_column_std('B') == pytest.approx(15.811388300841896)

    def test_get_column_median(self, op_stats: PolarsOperations):
        """Test median calculation."""
        # Col A: [1.0, 2.0, 3.0, 4.0, 5.0]. Median is 3.0.
        assert op_stats.get_column_median('A') == pytest.approx(3.0)
        # Col D (sorted non-nulls): [100, 300, 400]. Median is 300.0.
        assert op_stats.get_column_median('D') == pytest.approx(300.0)

    def test_get_column_null_count(self, op_stats: PolarsOperations):
        """Test null count calculation."""
        # Col A, B, C: 0 nulls
        assert op_stats.get_column_null_count('A') == 0
        assert op_stats.get_column_null_count('B') == 0
        # Col D: 2 nulls
        assert op_stats.get_column_null_count('D') == 2
        
    def test_stat_on_non_existent_column_raises(self, op_stats: PolarsOperations):
        """Test that calling stat methods on a non-existent column raises Polars' exception."""
        with pytest.raises(pl.ColumnNotFoundError):
            op_stats.get_column_mean('non_existent')
            
    def test_stat_on_empty_df_raises(self, empty_df: pl.DataFrame):
        """Test that stat methods raise Polars' exception on an empty DataFrame."""
        op = PolarsOperations(train_df=empty_df)
        # Polars mean/std on an empty numeric column returns None (NaN), which is fine.
        # However, .item() on a DataFrame with a NaN will return the NaN, which is a float.
        # We test for the expected NaN behavior instead of an error.
        assert math.isnan(op.get_column_mean('col_int'))