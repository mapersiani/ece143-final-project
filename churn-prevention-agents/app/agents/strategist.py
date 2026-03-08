import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts/strategist.txt")
_SYSTEM_PROMPT = open(_PROMPT_PATH).read()

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        model = os.getenv("GEMINI_STRATEGIST_MODEL", "gemini-2.5-flash")
        _llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
    return _llm


def run_strategist(analyst_report: dict, previous_critiques: list[dict]) -> dict:
    context = _build_context(analyst_report, previous_critiques)
    response = _get_llm().invoke([HumanMessage(content=f"{_SYSTEM_PROMPT}\n\n{context}")])

    # Parse JSON from LLM response
    return _parse_json(response.content)


def _build_context(analyst_report: dict, previous_critiques: list[dict]) -> str:
    ctx = f"ANALYST REPORT:\n{json.dumps(analyst_report, indent=2)}"

    if previous_critiques:
        last_critique = previous_critiques[-1]
        ctx += f"\n\nPREVIOUS CRITIQUE (Round {len(previous_critiques)}):\n{json.dumps(last_critique, indent=2)}"
        ctx += "\n\nRevise your proposals to address all CRITICAL objections above."
    else:
        ctx += "\n\nThis is Round 1. Propose initial retention strategies."

    return ctx


def _parse_json(content: str) -> dict:
    # Strip markdown code fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())
