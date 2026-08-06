# AGENTS.md

This file serves two purposes for the Bukka repository:
1. Contributor guidance for humans.
2. Runtime guidance for coding agents.

## Repository overview

- Project: `bukka`
- Language: Python (>=3.10)
- Package root: `/home/runner/work/Bukka/Bukka/src/bukka`
- Tests: `/home/runner/work/Bukka/Bukka/test`
- Docs: `/home/runner/work/Bukka/Bukka/docs`

## Contributor guide

### Local setup

1. Create and activate a virtual environment.
2. Install the project in editable mode with dev dependencies.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Run tests

Use pytest from repository root:

```bash
pytest
```

### Code style and scope

- Make focused, minimal changes for each task.
- Keep behavior backward-compatible unless the task requires otherwise.
- Update docs when behavior or CLI usage changes.
- Do not commit secrets, credentials, or generated local environment artifacts.

### Pull request and branching policy

Follow this branch flow strictly:

1. Create feature/fix branches from the `release` branch (not from `main`).
2. Open PRs targeting the `release` branch.
3. Do **not** merge feature/fix branches directly into `main`.
4. Only the `release` branch is merged into `main`.

This repository uses `release` as the integration gate before `main`.

## Runtime agent instructions

These rules apply to coding agents operating in this repository:

1. Prefer surgical edits and avoid unrelated refactors.
2. Inspect existing files before changing behavior.
3. Run existing test suite (`pytest`) after code changes.
4. Preserve project structure under `src/bukka` and `test`.
5. Avoid introducing new dependencies unless necessary.
6. Never commit secrets or sensitive data.
7. Keep PRs small, reviewable, and aligned with the branching policy above.
8. When uncertain about intent or scope, ask for clarification before broad changes.

## Paths reference

- Root: `/home/runner/work/Bukka/Bukka`
- Source: `/home/runner/work/Bukka/Bukka/src/bukka`
- Tests: `/home/runner/work/Bukka/Bukka/test`
- Docs: `/home/runner/work/Bukka/Bukka/docs`
