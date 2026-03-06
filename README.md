# ECE 143 Final Project

## Overview

Reducing customer churn is critical for e-commerce businesses. This project implements an integrated system that (1) predicts churn using ensemble ML methods and (2) segments customers for targeted interventions using AI Agents.

## Team

- Ayoob Al-Delaimy
- Allen Ekoung Keng
- Jason Kupai
- Narain Mylapore Sudhakar
- Matteo Persiani
- Raghusrinivasan Venkatesan

## Data sources

- **Olist Brazilian E-Commerce**  
  [Kaggle: olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)  
  Use: `olist_orders_dataset.csv`, `olist_customers_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`.

- **E-Commerce Customer Insights and Churn**  
  [Kaggle: nabihazahid/e-commerce-customer-insights-and-churn-dataset](https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset/data)  

## Research question

- Primary: Can we predict which customers are likely to churn and segment them for targeted interventions?
- Supporting: What are the main churn drivers (e.g. recency, satisfaction, tenure)?

## Repo structure

| Path | Purpose |
|------|--------|
| **`src/`** | Reserved for later: `data_pipeline.py`, `ml_models.py`, `agents.py` (reorganize from notebooks when done). |
| **`notebooks/`** | Analysis and pipeline code: 00 setup, 01 EDA, 02 features, 03 modeling, 04 segmentation, 05 agent demo. |
| **`data/raw/`** | Original CSVs |
| **`data/processed/`** | Final analysis-ready data (e.g. `churn_features.csv`). |
| **`outputs/`** | Model and pipeline outputs (not tracked): `stage1/` for churn models and high-risk segments, `stage2/` for agent-generated campaign JSON. |
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
5. Run notebooks in order: **00_project_setup** → **03_modeling** (and 01, 02, 04, 05 as needed). 00 writes `data/processed/churn_features.csv`; Stage 1 notebooks write `outputs/stage1/`; Stage 2 writes `outputs/stage2/`.
6. (Optional) Install pre-commit hooks:  
   `pre-commit install`  
   `pre-commit install --hook-type pre-push`
