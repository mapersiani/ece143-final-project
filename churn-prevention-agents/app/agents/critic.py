import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session
from app.db.memory import get_recent_actions, get_constraints

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts/critic.txt")
_SYSTEM_PROMPT = open(_PROMPT_PATH).read()

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        model = os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.5-pro")
        _llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
    return _llm


def run_critic(proposals: dict, analyst_report: dict, db: Session) -> dict:
    segment_ids = [s["segment_id"] for s in analyst_report.get("segments", [])]

    # Pull memory context for Critic to verify claims
    past_actions = get_recent_actions(db, segment_ids, limit=10)
    constraints = get_constraints(db)

    context = _build_context(proposals, past_actions, constraints)
    response = _get_llm().invoke([HumanMessage(content=f"{_SYSTEM_PROMPT}\n\n{context}")])

    return _parse_json(response.content)


def _build_context(proposals: dict, past_actions: list[dict], constraints: list[dict]) -> str:
    ctx = f"STRATEGIST PROPOSALS:\n{json.dumps(proposals, indent=2)}"
    ctx += f"\n\nHISTORICAL MEMORY (past campaign outcomes):\n{json.dumps(past_actions, indent=2)}"
    ctx += f"\n\nLEARNED CONSTRAINTS:\n{json.dumps(constraints, indent=2)}"
    return ctx


def _parse_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())
