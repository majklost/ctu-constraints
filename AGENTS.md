# Repository workflow

## Python environment

- Use the project virtual environment: `.venv/bin/python`.
- Run Python tooling with `.venv/bin/python`, `.venv/bin/ruff`, etc.
- Do not modify `.venv`, `uv.lock`, or dependencies unless the task asks for it.

## Jupytext notebooks

- `notebooks/**/*.py` in Jupytext `py:percent` format is the canonical,
  version-controlled notebook source.
- `notebooks/**/*.ipynb` is a local generated UI/output artifact. Do not edit,
  add, remove, or commit it as part of an ordinary code change.
- When asked to modify/refactor a notebook, edit its paired `.py` file—not its
  `.ipynb` file.
- After changing a paired notebook `.py`, run:
  `.venv/bin/jupytext --sync path/to/notebook.py`
  unless the user has the paired notebook open with unsaved changes.
