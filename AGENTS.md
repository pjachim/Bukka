# AGENTS.md

This file is the authoritative operating guide for contributors and coding agents working in this repository.

It defines:

1. How to use the `bukka` package safely.
2. How to update the package while preserving architecture, compatibility, and quality.

## Repository essentials

- Project: `bukka`
- Language: Python >= 3.10
- Source root: `src/bukka`
- Tests root: `test`
- Docs root: `docs`
- Main CLI entrypoint: `src/bukka/__main__.py`

## Branching and PR policy (required)

Follow this branch flow strictly:

1. Create feature/fix branches from the active `release/*` branch (for example, `release/1.0.2`), not from `main`.
2. Open PRs targeting that active `release/*` branch.
3. Do **not** merge feature/fix branches directly into `main`.
4. Only `release/*` branches are merged into `main`.

## Operating modes for agents

Use this section to decide the expected behavior before you start work.

### Mode A: Use the package (no source changes)

Choose this mode when the task is to run Bukka, generate outputs, or validate behavior.

1. Create and activate a virtual environment.
2. Install dependencies used in CI for local parity.
3. Run the CLI through `python -m bukka ...`.
4. Do not modify package source unless explicitly requested.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pytest polars pyarrow
python -m pip install -e .
```

Windows activation variants:

```powershell
.venv\Scripts\Activate.ps1
```

```cmd
.venv\Scripts\activate.bat
```

Common usage commands:

```bash
python -m bukka init-config
python -m bukka run --help
```

### Mode B: Update the package (source changes allowed)

Choose this mode when implementing features, bug fixes, refactors, or test/doc updates.

Required workflow:

1. Read affected modules before editing.
2. Make minimal, focused changes only.
3. Preserve public behavior unless change is explicitly requested.
4. Add or update tests with every behavior change.
5. Run relevant tests, then run broader tests as confidence check.
6. Update docs when CLI/config/output behavior changes.

## Architecture contracts agents must preserve

These are stability contracts. Do not change them casually.

1. `src/bukka/project.py` is the orchestration hub. Keep artifact generation delegated to dedicated writers in `src/bukka/coding/`.
2. `src/bukka/utils/files/file_manager.py` owns generated-project paths and skeleton creation. Avoid hardcoded path strings in other modules.
3. Generated train/test split artifacts must remain:
	- `data/train/train_data.pqt`
	- `data/test/test_data.pqt`
4. Generated config/reader code should use relative paths based on FileManager-derived values.
5. Narwhals is the dataframe boundary for reader/data flows; conversions to native frames should happen only at explicit edges.
6. The generated `DataReader` API differs by supervised vs target-less runs. Preserve that split unless intentionally redesigning and updating all dependent tests/docs.
7. New generated artifacts should be wired through:
	- FileManager path ownership
	- a dedicated writer in `src/bukka/coding/`
	- `Project.run()` orchestration

## Coding standards (required for updates)

1. Use Python 3.10+ type hints (`list[str]`, `str | None`, etc.).
2. Use NumPy-style docstrings with examples for public, non-trivial methods.
3. Prefer composition and explicit small helpers over monolithic methods.
4. Keep edits surgical; avoid unrelated renames/refactors.
5. Avoid adding dependencies unless clearly necessary.
6. Never commit secrets, credentials, or local environment artifacts.

## Testing standards (required for updates)

1. Use class-based pytest test classes for new tests.
2. Prefer targeted tests first, then full-suite validation when practical.
3. Avoid creating real virtual environments or running pip in most tests.
4. For CLI/integration tests, prefer `sys.executable` subprocess usage or `python -m bukka`.
5. When touching environment creation behavior, isolate via monkeypatching where possible unless the test is explicitly for real environment creation.

Recommended commands:

```bash
python -m pytest -v
python -m pytest test/test_data_management/test_dataset.py -v
python -m pytest test/test_data_management/test_dataset.py::TestDataset::test_init_sets_feature_columns_and_data_schema -v
python -m pytest -m venv -v
python -m build
python -m pip install -r docs/requirements.txt && python -m pip install -e . && cd docs && make html
```

## CLI and config coupling rules

When adding or changing a CLI/config option, keep these in sync:

1. `argparse` definitions in `src/bukka/cli_config.py`
2. `DEFAULT_CONFIG`
3. `BukkaConfig`
4. validation logic
5. `to_project_kwargs()` mapping

Do not update only one layer.

## Documentation and release hygiene

When behavior changes, update corresponding docs in `docs/` and relevant README/CLI examples.

Before proposing merge:

1. Confirm tests for changed behavior pass.
2. Confirm no unrelated file churn.
3. Confirm branch target follows `release/*` policy.
4. Summarize behavior impact and migration implications in PR notes.

## Agent execution checklist

Use this checklist every time you edit code:

1. Confirm task scope and non-goals.
2. Identify architecture contracts that apply.
3. Implement minimal code changes.
4. Add/update class-based pytest tests.
5. Run relevant tests and record results.
6. Update docs if external behavior changed.
7. Verify no secrets/artifacts were introduced.
8. Prepare concise change summary with risk notes.
