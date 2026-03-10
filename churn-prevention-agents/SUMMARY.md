# Churn Prevention Agents — Complete Summary

## Dataset: `customer_churn_features.csv`
- **Source**: `segmentation_and_evaluation` branch, `data/processed/`
- **Size**: 36,992 rows, 26 columns
- **Overall churn rate**: 54.1% (20,012 churned, 16,980 retained)

---

## Part 1: Exploratory Data Analysis

Generated **8 figures** saved in `figures/`:

| Figure | Key Finding |
|--------|------------|
| 01 - Churn Distribution | ~54/46 split, slightly skewed toward churn but well-balanced for ML |
| 02 - Churn by Membership | **Most powerful predictor.** Categories 0 & 2 have ~97% churn; categories 3 & 4 have 0% churn |
| 03 - Numeric Features | Retained customers have higher transaction values (median $30.6K vs $25.4K) and more wallet points (749 vs 648) |
| 04 - Correlation Heatmap | `membership_category` has strongest correlation with churn (-0.46). `support_risk` and `past_complaint` are perfectly correlated (r=1.0) — one is redundant |
| 05 - Segments | Low/medium value segments churn above 50%; high-value is below 50%. Engagement segmentation barely differentiates churn |
| 06 - Complaints & Feedback | Feedback categories 4-6, 8 have **0% churn**; categories 0-3, 7 have ~63-65% churn. Near-perfect signal |
| 07 - Tenure Distribution | Churn is spread uniformly across all tenure levels — not a lifecycle problem |
| 08 - Price Sensitivity | Marginal difference (<2 points) — not a meaningful churn driver |

### Key EDA Takeaways
1. Membership category and feedback type are near-deterministic churn signals
2. Monetary engagement (transaction value, wallet points) separates churners from retained
3. Demographics, recency, tenure, and price sensitivity are NOT useful predictors
4. `support_risk` = `past_complaint` (redundant, drop one)

---

## Part 2: XGBoost Model Training

Trained via the `POST /api/v1/train` endpoint inside Docker.

**Model**: XGBClassifier (300 trees, max_depth=6, learning_rate=0.1, subsample=0.8)

### Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 93.35% |
| Precision | 93.14% |
| Recall | 94.68% |
| F1-Score | 93.90% |
| AUC | **97.59%** |

### Top Feature Importances (confirming EDA)

| Feature | Importance |
|---------|-----------|
| `membership_category` | **46.4%** |
| `points_in_wallet` | **21.0%** |
| `feedback` | 4.3% |
| `avg_transaction_value` | 2.5% |
| All others | <1.5% each |

Model logged to **MLflow** and saved to `/app/models/churn_model.joblib` inside the container.

---

## Part 3: Agentic Pipeline (Full End-to-End)

Submitted the same dataset to `POST /api/v1/analyze` with the trained model backing the Analyst.

- **Pipeline flow**: Analyst → Strategist → Critic → Strategist (revised) → Critic (approved)
- **Duration**: ~58 seconds, 2 debate rounds

### Analyst Report (using real XGBoost predictions)

- **19,941 customers at risk** out of 36,992
- 2 segments identified by top churn driver:

| Segment | Count | Avg Churn Prob | Avg CLV | Priority Score | Root Cause |
|---------|-------|---------------|---------|----------------|------------|
| `driver_avg` | 18,056 | 95.2% | $16,637 | 15,836 | Transaction value patterns |
| `driver_points` | 1,885 | 95.0% | $1,972 | 1,874 | Loyalty points engagement |

- **Global top drivers**: membership, points, feedback, avg_transaction_value, tenure

### Debate Process

- **Round 1**: Strategist proposed email campaigns and discounts. Critic rated **3/10** and rejected — no measurement strategy, no KPIs, generic root cause unaddressed.
- **Round 2**: Strategist revised with validation-first approach, A/B testing, control groups. Critic rated **8/10** and **approved**.

### Approved Retention Plan

**`driver_avg` segment (HIGH priority, 18,056 customers)**:
1. Validate the 95.2% churn probability via customer outreach/survey to assess model calibration
2. A/B test: 10% discount for 3 months vs personalized feature highlight, measuring churn reduction against control group
3. Analyze qualitative feedback to identify specific pain points behind the generic "avg" driver

**`driver_points` segment (MEDIUM priority, 1,885 customers)**:
1. Validate 95% churn probability through targeted customer surveys
2. A/B test: personalized points redemption example ("You have X points, redeem for Y!") vs standard points reminder
3. Phased rollout starting with small subset to refine offers before scaling

### Risks Flagged by the Agents
- Churn probabilities above 95% suggest possible model calibration issues
- No historical data to benchmark against
- "avg" root cause needs decomposition into specific sub-drivers
- A/B tests need minimum sample sizes for statistical significance

---

## Code Changes

**Branch**: `agentic-pipeline-prompt`

### Modified
- `app/ml/train.py` — Real XGBoost training replacing mock placeholder
- `app/ml/model.py` — Trained model inference with mock fallback
- `app/agents/analyst.py` — CLV estimation for real dataset columns
- `app/api/routes.py` — New `POST /api/v1/train` endpoint
- `requirements.txt` — Added scikit-learn, xgboost, joblib

### Added
- `figures/` — 8 EDA plots
- `scripts/eda_plots.py` — EDA plotting script
- `sample_churn_data.csv` — Quick-test sample data

---

## Presentation Story Arc

```
EDA  →  "Here's what the data looks like; membership & feedback dominate churn"
  │
XGBoost  →  "Model confirms it: 97.6% AUC, membership=46%, points=21%"
  │
Agentic Pipeline  →  "But predictions alone aren't enough. How do we ACT?"
  │
Strategist proposes  →  Critic rejects (3/10, no measurement plan)
  │
Strategist revises  →  Critic approves (8/10, validation + A/B testing)
  │
Output: Actionable retention campaigns grounded in real ML insights
```
