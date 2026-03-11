# ECE 143 Final Project: Churn Prevention Agents

## Overview
Agentic churn prevention system using LangGraph debate arena, MLFlow experiment tracking, and GCP deployment. The system analyzes customer churn probability and uses multiple AI agents to propose, critique, and finalize retention strategies.

## Team
- Ayoob Al-Delamy
- Allen Ekoung Keng
- Jason Kupai
- Narain Mylapore Sudhakar
- Matteo Persiani
- Raghusrinivasan Venkatesan

## Data Source
- Customer Churn Dataset from HuggingFace - https://huggingface.co/datasets/d0r1h/customer_churn

## Architecture

Upload CSV → Analyst Agent → [Strategist ↔ Critic] Debate Arena → Executor Agent
                                     ↕
                              Memory Store (PostgreSQL)

## Project Structure
We have restructured the project so the implementation sits directly at the root.

```
app/                 # Main application source code
├── main.py          # FastAPI entrypoint
├── api/             # API Routes and schemas
├── agents/          # LangGraph nodes + prompts
│   ├── graph.py     # State machine
│   ├── analyst.py   # ML inference + segmentation
│   ├── strategist.py# Strategy proposals (Gemini)
│   ├── critic.py    # Critique + memory queries (Gemini)
│   └── executor.py  # Action generation + memory writes
└── db/              # Database models and memory management
    ├── models.py    # SQLAlchemy ORM
    └── memory.py    # Memory store queries
data/                # Raw and processed datasets (not tracked in git)
docker/              # Dockerfile + docker-compose for running locally
docs/                # Project proposal and presentation references
notebooks/           # Exploration and analysis notebooks
terraform/           # Infrastructure as Code for GCP deployment
tests/               # Unit tests
requirements.txt     # Python dependencies for the app
environment.yml      # Conda environment definition for notebooks
```

## Third-Party Modules
Our project relies on the following third-party dependencies:
- **FastAPI / Uvicorn**: Web framework and server for building and running the API.
- **Pandas / NumPy**: Data manipulation and numerical computations.
- **Scikit-learn / XGBoost**: Machine learning tools for training the churn prediction model.
- **Joblib**: Used for persisting and loading ML models.
- **MLflow**: Experiment tracking and model registry.
- **LangChain / LangGraph**: Frameworks to orchestrate our LLM agents and manage conversational state.
- **LangChain Google GenAI**: Integration required to use Google's Gemini LLMs.
- **Pydantic**: Data validation for our API request/response schemas.
- **SQLAlchemy / Psycopg2**: ORM and database driver for interacting with PostgreSQL.
- **Python-dotenv**: Used for loading environment variables.

## How to Run the Code

### 1. Local Development (with Docker)
This is the recommended way to test the full pipeline (API + PostgreSQL + MLFlow) locally.

1. **Set your Gemini API key**:
   ```bash
   cp .env.example .env
   # Edit .env and set GOOGLE_API_KEY inside.
   ```
2. **Start the stack**:
   ```bash
   cd docker
   docker compose up --build
   ```
   - **App API**: http://localhost:8000
   - **MLFlow UI**: http://localhost:5000
   - **API Docs**: http://localhost:8000/docs

3. **Run the Pipeline**:
   ```bash
   # Upload a CSV and start analysis
   curl -X POST http://localhost:8000/api/v1/analyze \
     -F "file=@your_churn_data.csv"
   
   # Poll for results using the returned job_id
   curl http://localhost:8000/api/v1/results/{job_id}
   ```

### 2. Notebooks & Data Exploration (Conda)
If you only want to explore the data using Jupyter notebooks:
1. **Create the environment**:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the environment**:
   ```bash
   conda activate ece143-final
   ```
3. **Start Jupyter**:
   ```bash
   jupyter lab
   ```

## Workflow 
- Use notebooks for exploration and prototyping.
- Keep data under `data/` and do not commit raw datasets.
- Send requests to the `/analyze` endpoint after inferring results from the trained model.
- Poll the `/results` endpoint for getting the insights from the model.
