from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.api.routes import router
from app.db.memory import init_db

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Churn Prevention Agents",
    description="Agentic churn prevention pipeline with LangGraph debate arena",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
