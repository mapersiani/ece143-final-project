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

## Run the Pipeline

```bash
# Upload a CSV and start analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@your_churn_data.csv"

# Returns: {"job_id": "...", "status": "processing"}

# Poll for results
curl http://localhost:8000/api/v1/results/{job_id}
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
└── db/
    ├── models.py        # SQLAlchemy ORM
    └── memory.py        # Memory store queries

docker/                  # Dockerfile + docker-compose
tests/                   # Unit tests
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://churn_user:churn_pass@localhost:5432/churn_db` |
| `GOOGLE_API_KEY` | Gemini API key | required |
| `MAX_DEBATE_ROUNDS` | Max debate rounds before escalation | `5` |
| `CONSENSUS_THRESHOLD` | Critic rating threshold (1-10) for approval | `7` |
