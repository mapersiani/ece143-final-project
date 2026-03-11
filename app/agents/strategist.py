import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts/strategist.txt")
_SYSTEM_PROMPT = open(_PROMPT_PATH).read()

_llm = None


def _get_llm():
    """
    Lazily construct and cache the strategist language model client.

    :return: A configured `ChatGoogleGenerativeAI` instance for strategist prompts.
    """
    global _llm
    if _llm is None:
        model = os.getenv("GEMINI_STRATEGIST_MODEL", "gemini-2.5-flash")
        _llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
    return _llm


def run_strategist(analyst_report: dict, previous_critiques: list[dict]) -> dict:
    """
    Generate updated retention strategy proposals based on the analyst report and prior critiques.

    :param analyst_report: Structured summary of at-risk segments and churn drivers.
    :param previous_critiques: List of previous critic responses used to refine proposals.
    :return: A dictionary of strategist proposals parsed from the LLM response.
    """
    context = _build_context(analyst_report, previous_critiques)
    response = _get_llm().invoke([HumanMessage(content=f"{_SYSTEM_PROMPT}\n\n{context}")])

    # Parse JSON from LLM response
    return _parse_json(response.content)


def _build_context(analyst_report: dict, previous_critiques: list[dict]) -> str:
    """
    Build the concatenated text context fed into the strategist model.

    :param analyst_report: Structured analyst output describing risk segments and drivers.
    :param previous_critiques: Historical critic feedback used to inform revisions.
    :return: A formatted string containing the analyst report and latest critique context.
    """
    ctx = f"ANALYST REPORT:\n{json.dumps(analyst_report, indent=2)}"

    if previous_critiques:
        last_critique = previous_critiques[-1]
        ctx += f"\n\nPREVIOUS CRITIQUE (Round {len(previous_critiques)}):\n{json.dumps(last_critique, indent=2)}"
        ctx += "\n\nRevise your proposals to address all CRITICAL objections above."
    else:
        ctx += "\n\nThis is Round 1. Propose initial retention strategies."

    return ctx


def _parse_json(content: str) -> dict:
    """
    Parse JSON strategist output, handling optional Markdown code fences.

    :param content: Raw text content returned by the LLM.
    :return: Parsed JSON object representing strategist proposals.
    """
    # Strip markdown code fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())
