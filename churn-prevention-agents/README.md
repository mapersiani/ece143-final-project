# E-Commerce Customer Churn Prevention

An end-to-end system that combines ML-based churn prediction with an agentic AI pipeline to automatically generate, debate, and refine actionable customer retention strategies.

## File Structure

```
churn-prevention-agents/
├── README.md                       # This file
├── SUMMARY.md                      # Detailed project summary and findings
├── requirements.txt                # Python dependencies
├── data/
│   └── customer_churn_features.csv # Main dataset (36,992 rows, 26 columns)
├── src/                            # Modular Python source code
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and validation utilities
│   ├── preprocessing.py            # Feature engineering, encoding, stats
│   ├── eda.py                      # 8 EDA plotting functions
│   └── model_training.py           # XGBoost training, evaluation, feature importance
├── notebooks/
│   └── eda_and_model.ipynb         # Jupyter notebook with all visualizations
├── figures/                        # Generated EDA plots (PNG)
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_membership.png
│   ├── 03_numeric_by_churn.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_churn_by_segments.png
│   ├── 06_complaint_support_churn.png
│   ├── 07_tenure_distribution.png
│   └── 08_churn_by_price_sensitivity.png
├── scripts/
│   ├── eda_plots.py                # Standalone script to regenerate all figures
│   └── enhance_pptx.py             # Script to add slides to the presentation
├── app/                            # FastAPI agentic pipeline application
│   ├── main.py                     # App entrypoint
│   ├── agents/                     # LangGraph agent implementations
│   │   ├── analyst.py              # Churn analysis and segmentation
│   │   ├── strategist.py           # Retention strategy generation (Gemini)
│   │   ├── critic.py               # Strategy evaluation and scoring (Gemini)
│   │   ├── executor.py             # Campaign package generation (Gemini)
│   │   ├── graph.py                # LangGraph state machine orchestration
│   │   └── prompts/                # Agent system prompts
│   │       ├── strategist.txt
│   │       ├── critic.txt
│   │       └── executor.txt
│   ├── api/                        # FastAPI routes and schemas
│   │   ├── routes.py               # API endpoints (/train, /analyze, /results, etc.)
│   │   └── schemas.py              # Pydantic request/response models
│   ├── db/                         # Database layer
│   │   ├── memory.py               # CRUD operations
│   │   └── models.py               # SQLAlchemy ORM models
│   └── ml/                         # ML model layer (used inside Docker)
│       ├── model.py                # Inference with trained model or mock fallback
│       └── train.py                # XGBoost training with MLflow logging
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml          # Postgres + MLflow + App services
├── terraform/                      # GCP deployment (Cloud Run + Cloud SQL)
└── tests/
    └── test_agents.py
```

## How to Run

### 1. Run the Jupyter Notebook (EDA + Model Training)

```bash
cd churn-prevention-agents

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/eda_and_model.ipynb
```

The notebook imports from `src/` and generates all 8 EDA visualizations plus XGBoost model training with metrics and feature importance plots.

### 2. Regenerate EDA Figures Only

```bash
cd churn-prevention-agents
python scripts/eda_plots.py
```

Saves all 8 PNG figures to the `figures/` directory.

### 3. Run the Agentic Pipeline (Docker)

```bash
cd churn-prevention-agents

# Copy and configure environment variables
cp .env.example .env
# Edit .env to add your GOOGLE_API_KEY (Gemini)

# Start all services
cd docker
docker compose up --build -d

# Train the model
curl -X POST http://localhost:8000/api/v1/train \
  -F "file=@../data/customer_churn_features.csv" \
  -F "target_col=churn_risk_score"

# Run the full pipeline
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@../data/customer_churn_features.csv"

# Check status (replace JOB_ID)
curl http://localhost:8000/api/v1/status/{JOB_ID}

# Get results
curl http://localhost:8000/api/v1/results/{JOB_ID}
```

### API Endpoints

| Method | Endpoint             | Description                              |
|--------|----------------------|------------------------------------------|
| GET    | `/health`            | Health check                             |
| POST   | `/api/v1/train`      | Train XGBoost model on uploaded CSV      |
| POST   | `/api/v1/analyze`    | Run full agentic pipeline on uploaded CSV|
| GET    | `/api/v1/status/{id}`| Check pipeline job status                |
| GET    | `/api/v1/results/{id}`| Get full pipeline results               |
| POST   | `/api/v1/feedback`   | Submit outcome feedback for actions      |
| GET    | `/api/v1/models`     | List available Gemini models             |
| GET    | `/api/v1/experiments`| List MLflow experiments and runs         |

## Third-Party Modules

| Module | Version | Purpose |
|--------|---------|---------|
| pandas | >= 2.2 | Data manipulation and analysis |
| numpy | >= 1.26 | Numerical computing |
| matplotlib | >= 3.8 | Plotting and visualization |
| seaborn | >= 0.13 | Statistical visualization |
| scikit-learn | >= 1.4 | ML utilities (train/test split, metrics) |
| xgboost | >= 2.0 | Gradient boosting classifier |
| joblib | >= 1.3 | Model serialization |
| fastapi | >= 0.110 | Web framework for API |
| uvicorn | >= 0.29 | ASGI server |
| sqlalchemy | >= 2.0 | ORM for PostgreSQL |
| psycopg2-binary | >= 2.9 | PostgreSQL adapter |
| mlflow | >= 2.12 | Experiment tracking |
| langchain-google-genai | >= 1.0 | Gemini LLM integration |
| langgraph | >= 0.2 | Agent orchestration framework |
| google-generativeai | >= 0.7 | Google Generative AI SDK |
| python-dotenv | >= 1.0 | Environment variable loading |
| python-multipart | >= 0.0.9 | File upload handling |
| pydantic | >= 2.7 | Data validation |
| python-pptx | >= 1.0 | PowerPoint generation (scripts only) |

## Key Results

- **XGBoost AUC: 97.6%** on 36,992 customer records
- **Top features:** membership_category (46.4%), points_in_wallet (21.0%), feedback (4.3%)
- **Agentic pipeline:** Strategist-Critic debate improved strategy from 3/10 to 8/10 rating
- **19,941 at-risk customers** identified and segmented with actionable retention plans
