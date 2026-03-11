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
- Customer Churn Dataset from HuggingFace - (link)

## Research Question
- Factors driving customer churn and personalized retention strategy generation

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
- `churn-prevention-agents/`: Agentic churn prevention app (see below)

### Churn Prevention Agents (`churn-prevention-agents/`) — file structure

```
churn-prevention-agents/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/
│   │   ├── routes.py           # Endpoints: /analyze, /results, /status, /models, /feedback, /experiments
│   │   └── schemas.py          # Pydantic request/response models
│   ├── agents/
│   │   ├── graph.py            # LangGraph state machine (analyst → strategist ↔ critic → executor)
│   │   ├── analyst.py          # Segmentation + memory lookup
│   │   ├── strategist.py       # Strategy proposals (Gemini)
│   │   ├── critic.py           # Critique + memory queries (Gemini)
│   │   ├── executor.py        # Campaign generation + memory writes (Gemini)
│   │   └── prompts/
│   │       ├── strategist.txt
│   │       ├── critic.txt
│   │       └── executor.txt
│   └── db/
│       ├── models.py           # SQLAlchemy: pipeline_runs, actions_history, segment_profiles, constraints
│       └── memory.py           # Memory store read/write
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml      # app + postgres + mlflow
├── tests/
│   └── test_agents.py
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
└── README.md
```

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
- Send requests to the `/analyze` endpoint after inferring results from the trained model
- Poll the `/results` endpoint for getting the insights from the model
