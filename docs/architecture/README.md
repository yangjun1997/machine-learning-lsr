# Architecture Overview

## Reading Order

1. `README.md` — repository purpose and quick start.
2. `PIPELINE.md` — numbered scripts, outputs, and dependencies.
3. `CONVENTIONS.md` — naming and synchronization rules.

## Current Data Flow

The package in `src/mechine_learning_lsr/` provides reusable Python code and the
`mechine-learning-lsr` console entry point. Future analysis scripts in
`scripts_py/` should read external or documented inputs and write their own
`results/NN_description/` directory. Shared logic belongs in `src/`; scripts
should stay thin and reproducible.

This project currently has no research-version history or committed datasets.
Add versioned hypothesis documents only when the work becomes hypothesis-driven
research.
