# Repository Guidelines

## Project Structure & Module Organization

- `src/mechine_learning_lsr/` contains the installable Python package. Keep package code under this `src/` layout.
- `src/mechine_learning_lsr/__init__.py` currently exposes the `main` console entry point.
- `openspec/` contains spec-driven change artifacts and project configuration; keep proposals and implementation notes there when a change requires them.
- `README.md` is the user-facing overview. Add tests under `test/` as behavior is introduced.

## Build, Test, and Development Commands

Use Python 3.12 or newer and manage the environment with `uv`:

```powershell
uv sync                         # Create/update the environment
uv run mechine-learning-lsr    # Run the console entry point
uv run python -m pytest         # Run tests (when test/ exists)
uv build                        # Build wheel and source distribution
```

Run commands from the repository root. Keep generated build output (for example, `dist/`) out of commits.

## Coding Style & Naming Conventions

Follow standard Python formatting: 4 spaces, no tabs, readable lines, and type hints for public functions. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Prefer the standard library and small, direct functions. If formatting or linting tools are added, run them through `uv run` and document the command here.

## Testing Guidelines

Use `pytest` for automated tests, with files named `test_*.py` and test functions named `test_<behavior>`. Place tests in `test/`, mirroring package areas when the codebase grows. Every new non-trivial behavior should have a focused regression or unit test; run the full suite before opening a pull request.

## Commit & Pull Request Guidelines

This repository has no commits yet, so no established message convention is available. Use concise imperative subjects (for example, `Add input validation`) and keep each commit focused. Pull requests should explain the behavior change, identify relevant files or OpenSpec artifacts, include test/build results, and add screenshots only when user-visible output changes. Keep unrelated cleanup out of the same PR.

## Security & Configuration Tips

Do not commit secrets, local virtual environments, caches, or generated distributions. Review dependency and configuration changes in `pyproject.toml` before merging.
