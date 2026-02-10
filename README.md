# ECE 143 Final Project

## Overview
TBD: One-paragraph summary of the project, data source, and the main question.

## Team
- Ayoob Al-Delamy
- Allen Ekoung Keng
- Jason Kupai
- Narain Mylapore Sudhakar
- Matteo Persiani
- Raghusrinivasan Venkatesan

## Data Source
- TBD (public dataset link and access notes)

## Research Question
- TBD (primary question and any supporting sub-questions)

## Repo Structure
- `src/`: Reusable Python modules and utilities
- `notebooks/`: Exploration and analysis notebooks
- `data/`: Raw and processed datasets (not tracked in git)
  - `raw/`: Original data as obtained
  - `interim/`: Intermediate, cleaned, or transformed data
  - `processed/`: Final analysis-ready datasets
- `reports/figures/`: Exported plots and figures for the presentation
- `scripts/`: One-off scripts for data download, cleaning, or preprocessing
- `docs/`: Proposal and presentation references

## Quickstart (Conda)
1. Create environment:
   - `conda env create -f environment.yml`
2. Activate environment:
   - `conda activate ece143-final`
3. Start Jupyter:
   - `jupyter lab`
4. Install pre-commit hooks:
   - `pre-commit install`
   - `pre-commit install --hook-type pre-push`

## Workflow 
- Use notebooks for exploration and prototyping.
- Move reusable code into `src/` as it stabilizes.
- Keep data under `data/` and do not commit raw datasets.
- Run tests with `pytest` before opening PRs.
