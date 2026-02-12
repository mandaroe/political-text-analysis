# ML Class Project Template

This repo is set up as a class project template derived from a research project codebase. It keeps a realistic structure (data, scripts, configs, training code) while adding guidance for scoping, experimentation, and reporting.

If you want to refer to the `README` in the original codebase, see `docs/README_NEDS.md`.

## Quick Start (Students)

1. Read the project brief and fill in your scope: `docs/PROJECT_TEMPLATE.md`.
2. Set up the environment. **Replace the environment name and Python package dependencies with your own.**
   ```bash
   conda env create -f env.yaml
   conda activate YOUR_ENV_NAME
   ```

## Recommended Repo Layout

Use the existing folder structure, but **replace it with your own files**:

- `src/`: core model, training, evaluation, and utilities.
- `script/`: run scripts that record exact command-line commands when running on a cluster.
- `data/`: small metadata files or pointers to datasets.
- `assets/`: figures for the final report.
- `docs/`: project brief, milestones, and writeups.

## What Students Must Deliver

- A clearly scoped research question and hypothesis.
- A model implementation, with corresponding training and evaluation code.
- A reproducible experiment log with metrics and plots.

## Instructor Notes

- The training/eval scripts in the original codebase are intact for your reference.
- Suggested workflows are outlined in `docs/PROJECT_TEMPLATE.md`.


