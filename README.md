# ECE 143 Final Project

## Overview

Reducing customer churn is critical for e-commerce businesses. This project implements an integrated system that (1) predicts churn using ensemble ML methods and (2) segments customers for targeted interventions. **For now all logic lives in the notebooks** for easier setup and reading; you can reorganize into `src/` and a formal `data/` layout when the project is complete.

## Team

- Ayoob Al-Delaimy
- Allen Ekoung Keng
- Jason Kupai
- Narain Mylapore Sudhakar
- Matteo Persiani
- Raghusrinivasan Venkatesan

## Data sources

Put raw CSVs in **`data/raw/`**. All data (raw, interim, processed) lives under **`data/`** and is committed.

- **Olist Brazilian E-Commerce**  
  [Kaggle: olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)  
  Use: `olist_orders_dataset.csv`, `olist_customers_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`.

- **E-Commerce Customer Insights and Churn**  
  [Kaggle: nabihazahid/e-commerce-customer-insights-and-churn-dataset](https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset/data)  
  Use: `ECommerceAndChurnDataset.csv`.

See **`scripts/README.md`** for download instructions.

## Research question

- Primary: Can we predict which customers are likely to churn and segment them for targeted interventions?
- Supporting: What are the main churn drivers (e.g. recency, satisfaction, tenure)?

## Repo structure

| Path | Purpose |
|------|--------|
| **`src/`** | Reserved for later: `data_pipeline.py`, `ml_models.py`, `agents.py` (reorganize from notebooks when done). |
| **`notebooks/`** | Analysis and pipeline code: 00 setup, 01 EDA, 02 features, 03 modeling, 04 segmentation, 05 agent demo. |
| **`data/`** | All data (raw, interim, processed); committed. |
| **`data/raw/`** | Original CSVs from Kaggle. |
| **`data/interim/`** | Intermediate, cleaned, or transformed data. |
| **`data/processed/`** | Final analysis-ready data (e.g. `churn_features.csv`). |
| **`outputs/`** | Model and pipeline outputs: `high_risk_customers.csv`, metrics, etc. (not tracked). |
| **`reports/figures/`** | Exported plots for the report/presentation. |
| **`scripts/`** | One-off scripts: data setup, download, preprocessing helpers. |
| **`docs/`** | Proposal and presentation references. |
| **`tests/`** | Pytest tests. |

## Quickstart (Conda)

1. Create environment:  
   `conda env create -f environment.yml`
2. Activate:  
   `conda activate ece143-final`
3. **Data:** Put the required CSVs in **`data/raw/`** (see [Data sources](#data-sources)).
4. Start Jupyter from the **project root**:  
   `jupyter lab`
5. Run notebooks in order: **00_project_setup** → **03_modeling** (and 01, 02, 04, 05 as needed). 00 writes `data/processed/churn_features.csv`; 03 writes `outputs/`.
6. (Optional) Install pre-commit hooks:  
   `pre-commit install`  
   `pre-commit install --hook-type pre-push`

## Workflow

- Keep logic in notebooks for now; reorganize into `src/` when the project is complete.
- All data in `data/` (raw, interim, processed); model outputs in `outputs/`.
- Run tests with `pytest` when you have the conda env active.
