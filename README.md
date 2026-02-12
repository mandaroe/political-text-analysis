# ML Class Project Template (Based on NEDS)

This repo is set up as a class-ready project template derived from the NEDS research codebase. It keeps a realistic structure (data, scripts, configs, training code) while adding student-facing guidance for scoping, experimentation, and reporting.

If you want the original NEDS research README, see `docs/README_NEDS.md`.

## Quick Start (Students)

1. Read the project brief and fill in your scope: `docs/PROJECT_TEMPLATE.md`.
2. Set up the environment:
   ```bash
   conda env create -f env.yaml
   conda activate neds
   ```
3. Add your data or update `data/` with links/instructions.
4. Start from a baseline in `src/` and log all runs via `script/`.

## Recommended Repo Layout

Use the existing folders and add to them:

- `src/`: core model, training, evaluation, and utilities.
- `script/`: run scripts that record exact commands.
- `data/`: small metadata files or pointers to datasets.
- `assets/`: figures for the final report.
- `docs/`: project brief, milestones, and writeups.

## What Students Must Deliver

- A clearly scoped research question and hypothesis.
- A baseline implementation and at least one improved variant.
- A reproducible experiment log with metrics and plots.
- A short report and a demo (notebook or script).

## Instructor Notes

- The original NEDS training/eval scripts are intact for realism.
- You can swap datasets by updating `data/` and the relevant loaders in `src/`.
- Suggested milestones and grading rubric are in `docs/PROJECT_TEMPLATE.md`.


