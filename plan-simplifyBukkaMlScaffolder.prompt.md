# Plan: Simplify Bukka ML Scaffolder Codebase

This plan addresses unused code, over-engineered abstractions, and dead features to reduce maintenance burden while preserving extensibility.

## ✅ Completed Steps

1. ~~**Delete unused wrapper abstraction** in `src/bukka/data_management/wrapper/`~~ ✅ **COMPLETED**
   - The wrapper directory was empty and vestigial (polars.py never existed; the codebase has already migrated to Narwhals)
   - Deleted `src/bukka/data_management/wrapper/` entirely
   - Note: `src/bukka/utils/files/` is NOT a duplicate - it contains `FileManager` which is actively used throughout the codebase

3. ~~**Implement stratification** in `src/bukka/data_management/dataset_functionality/management.py`~~ ✅ **COMPLETED**
   - Implemented full stratified sampling in `split_dataset` method
   - When `stratify=True` (default), splits each group formed by `strata` columns (or `target_column` if strata is None) proportionally
   - When `stratify=False`, performs simple random split
   - Added intelligent fallback: if >50% of values are unique (e.g., continuous target), automatically falls back to non-stratified split
   - Handles edge cases: single-row groups are randomly assigned; ensures at least 1 sample per split when possible
   - Updated docstring with comprehensive parameter documentation and examples

4. ~~**Prune unused expert_system solutions**~~ ✅ **COMPLETED**
   - Connected text solutions (TfidfVectorizer, CountVectorizer, HashingVectorizer) to ProblemIdentifier
   - Added `is_text_column()` method to `DatasetQuality` and `Dataset` to detect text data (avg string length ≥ 50 chars)
   - Updated `_identify_univariate_problems()` to detect text columns and add text preprocessing solutions
   - Text solutions are now automatically suggested for columns with long text content

6. ~~**Fix dead preprocessing references**~~ ✅ **COMPLETED**
   - Created `src/bukka/preprocessing/` module with `categorical.py`
   - Implemented `standardize_categories` transformer: normalizes case, strips whitespace, applies custom mappings
   - Implemented `encode_categories` transformer: supports ordinal and label encoding
   - Both transformers follow scikit-learn's BaseEstimator/TransformerMixin pattern
   - Added comprehensive docstrings with numpy-style documentation and usage examples

## Test Results

All 238 tests pass ✅

## Further Considerations

1. **Backend parameter**: The `backend` parameter flows through CLI → Project → Dataset but only Polars is implemented. Finish switching from polars to narwhals to enable multiple backends.

3. **`expected_args` in Solution class**: Parameter is stored but never validated — implement validation for pipeline correctnes.
