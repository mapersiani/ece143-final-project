# Churn Prevention Agents

Agentic churn prevention system using LangGraph debate arena, MLFlow experiment tracking, and GCP deployment.

## Architecture

```
Upload CSV → Analyst Agent → [Strategist ↔ Critic] Debate Arena → Executor Agent
                                     ↕
                              Memory Store (PostgreSQL)
```

## Quick Start (Local with Docker)

```bash
# 1. Set your Gemini API key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY

# 2. Start the full stack
cd docker
docker compose up --build

# Services:
# App API:    http://localhost:8000
# MLFlow UI:  http://localhost:5000
# API Docs:   http://localhost:8000/docs
```

## Train the Churn Model

```bash
# From inside the app container or locally with deps installed
python -m app.ml.train
# Right now this is not implemented as we have not finalized our model and training pipeline yet.
# Model saved to models/churn_model.joblib
# Run tracked in MLFlow at http://localhost:5000
```

## Run the Pipeline

```bash
# Upload a CSV and start analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@your_churn_data.csv"

# Returns: {"job_id": "...", "status": "processing"}

# Poll for results
curl http://localhost:8000/api/v1/results/{job_id}
```

## Run Tests

```bash
pip install pytest
pytest tests/
```

## Deploy to GCP

```bash
# 1. Build and push Docker image
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/churn-prevention/churn-prevention-app:latest -f docker/Dockerfile .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/churn-prevention/churn-prevention-app:latest

# 2. Deploy infrastructure
cd terraform
terraform init
terraform apply \
  -var="project_id=YOUR_PROJECT" \
  -var="db_password=your-secure-password" \
  -var="google_api_key=your-gemini-key"

# 3. Outputs will show the app and MLFlow URLs
```

## Project Structure

```
app/
├── main.py              # FastAPI entrypoint
├── api/                 # Routes and schemas
├── agents/              # LangGraph nodes + prompts
│   ├── graph.py         # State machine
│   ├── analyst.py       # ML inference + segmentation
│   ├── strategist.py    # Strategy proposals (Gemini)
│   ├── critic.py        # Critique + memory queries (Gemini)
│   └── executor.py      # Action generation + memory writes
├── ml/
│   ├── train.py         # MLFlow-tracked XGBoost training
│   └── model.py         # Inference + SHAP
└── db/
    ├── models.py        # SQLAlchemy ORM
    └── memory.py        # Memory store queries

docker/                  # Dockerfile + docker-compose
terraform/               # GCP infrastructure
tests/                   # Unit tests
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://churn_user:churn_pass@localhost:5432/churn_db` |
| `MLFLOW_TRACKING_URI` | MLFlow server URL | `http://localhost:5000` |
| `GOOGLE_API_KEY` | Gemini API key | required |
| `MAX_DEBATE_ROUNDS` | Max debate rounds before escalation | `5` |
| `CONSENSUS_THRESHOLD` | Critic rating threshold (1-10) for approval | `7` |
