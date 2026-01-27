# Plan: Simplify Bukka ML Scaffolder Codebase

This plan addresses unused code, over-engineered abstractions, and dead features to reduce maintenance burden while preserving extensibility.

## Steps

1. **Delete unused wrapper abstraction** in `src/bukka/data_management/wrapper/` — the entire directory except `polars.py` appears vestigial; also remove duplicate `src/bukka/utils/files/` if it duplicates `FileManager` in logistics.

3. **Implement stratification** in `src/bukka/data_management/wrapper/polars.py` — `split_dataset` accepts `strata`/`stratify` parameters but ignores them; implement stratified sampling.

4. **Prune unused expert_system solutions** — multiple solution classes in `implemented_solutions/` (e.g., `TextPreprocessing`, `TfidfVectorizerSolution`, many regression/classification alternatives) are never wired to `ProblemIdentifier`; connect them.

6. **Fix or remove dead preprocessing references** in `src/bukka/expert_system/implemented_solutions/` — solutions reference `bukka.preprocessing.categorical` which doesn't exist; create the module.

## Further Considerations

1. **Backend parameter**: The `backend` parameter flows through CLI → Project → Dataset but only Polars is implemented. Finish switching from polars to narwhals to enable multiple backends.

3. **`expected_args` in Solution class**: Parameter is stored but never validated — implement validation for pipeline correctnes.
