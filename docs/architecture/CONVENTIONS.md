# Project Conventions

## Numbered Analysis Stages

- Use `scripts_py/NN_description.py` with a two-digit stage number.
- Write generated artifacts to `results/NN_description/`.
- Keep reusable code in `src/mechine_learning_lsr/`, not duplicated in scripts.

## Change Checklist

For a new analysis stage:

1. Define its inputs, outputs, and acceptance conditions in an OpenSpec change
   when the work is more than a small local edit.
2. Add the script and matching result directory.
3. Add or update a focused test under `test/`.
4. Add one row to `PIPELINE.md` and update this document if a convention changes.

Do not commit raw/private data, large generated results, virtual environments,
or tool caches.
