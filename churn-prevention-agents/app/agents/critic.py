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
    """
    Lazily construct and cache the critic language model client.

    :return: A configured `ChatGoogleGenerativeAI` instance for critique generation.
    """
    global _llm
    if _llm is None:
        model = os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.5-pro")
        _llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
    return _llm


def run_critic(proposals: dict, analyst_report: dict, db: Session) -> dict:
    """
    Critically evaluate strategist proposals using analyst context and historical memory.

    :param proposals: The latest strategist proposals to review.
    :param analyst_report: Analyst summary providing churn segments and drivers.
    :param db: Database session used to fetch past actions and constraints.
    :return: A structured critique dictionary parsed from the LLM response.
    """
    segment_ids = [s["segment_id"] for s in analyst_report.get("segments", [])]

    # Pull memory context for Critic to verify claims
    past_actions = get_recent_actions(db, segment_ids, limit=10)
    constraints = get_constraints(db)

    context = _build_context(proposals, past_actions, constraints)
    response = _get_llm().invoke([HumanMessage(content=f"{_SYSTEM_PROMPT}\n\n{context}")])

    return _parse_json(response.content)


def _build_context(proposals: dict, past_actions: list[dict], constraints: list[dict]) -> str:
    """
    Build the textual context presented to the critic model.

    :param proposals: Strategist proposals under review.
    :param past_actions: Recent campaign outcomes used as empirical evidence.
    :param constraints: Learned business constraints the critic must enforce.
    :return: A formatted string that combines proposals, history, and constraints.
    """
    ctx = f"STRATEGIST PROPOSALS:\n{json.dumps(proposals, indent=2)}"
    ctx += f"\n\nHISTORICAL MEMORY (past campaign outcomes):\n{json.dumps(past_actions, indent=2)}"
    ctx += f"\n\nLEARNED CONSTRAINTS:\n{json.dumps(constraints, indent=2)}"
    return ctx


def _parse_json(content: str) -> dict:
    """
    Parse JSON critic output, handling optional Markdown code fences.

    :param content: Raw text content returned by the LLM.
    :return: Parsed JSON object representing the critic's assessment.
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())
