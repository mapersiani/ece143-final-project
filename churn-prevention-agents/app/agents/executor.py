import json
import logging
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session
from app.db.memory import write_action

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts/executor.txt")
_SYSTEM_PROMPT = open(_PROMPT_PATH).read()

MAX_RETRIES = 3

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        model = os.getenv("GEMINI_EXECUTOR_MODEL", "gemini-2.0-flash-lite")
        _llm = ChatGoogleGenerativeAI(model=model, temperature=0.4)
    return _llm


def run_executor(approved_plan: dict, analyst_report: dict, db: Session, job_id: str) -> dict:
    context = f"APPROVED PLAN:\n{json.dumps(approved_plan, indent=2)}"
    context += f"\n\nANALYST REPORT CONTEXT:\n{json.dumps(analyst_report, indent=2)}"

    execution_package = _invoke_with_retry(context, job_id)

    for campaign in execution_package.get("campaigns", []):
        write_action(db, {
            "job_id": job_id,
            "segment_id": campaign["segment_id"],
            "action_type": campaign["action_type"],
            "action_details": campaign,
            "customer_count": _get_segment_count(analyst_report, campaign["segment_id"]),
            "status": "pending",
        })

    execution_package["created_at"] = datetime.utcnow().isoformat()
    return execution_package


def _invoke_with_retry(context: str, job_id: str) -> dict:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _get_llm().invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ])
            result = _parse_json(response.content)
            _validate_schema(result)
            logger.info(f"[{job_id}] Executor JSON parsed successfully on attempt {attempt}")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"[{job_id}] Executor attempt {attempt}/{MAX_RETRIES} invalid JSON: {e}")
            # Append correction hint to context for next attempt
            context += f"\n\nPREVIOUS ATTEMPT FAILED: {e}. Return only valid JSON matching the schema exactly."
        except Exception as e:
            # Non-JSON errors (network, API) — re-raise immediately
            raise

    raise ValueError(f"Executor failed to return valid JSON after {MAX_RETRIES} attempts: {last_error}")


def _validate_schema(data: dict):
    # Enforce required top-level keys and campaign field presence
    if "campaigns" not in data:
        raise ValueError("Missing required key: 'campaigns'")
    if not isinstance(data["campaigns"], list):
        raise ValueError("'campaigns' must be a list")
    required_fields = {"segment_id", "action_type", "subject_line", "message_template",
                       "success_metric", "control_pct", "treatment_pct", "review_after_days"}
    for i, campaign in enumerate(data["campaigns"]):
        missing = required_fields - campaign.keys()
        if missing:
            raise ValueError(f"Campaign[{i}] missing fields: {missing}")
        if campaign["control_pct"] + campaign["treatment_pct"] != 100:
            raise ValueError(f"Campaign[{i}] control_pct + treatment_pct must equal 100")


def _get_segment_count(analyst_report: dict, segment_id: str) -> int:
    for seg in analyst_report.get("segments", []):
        if seg["segment_id"] == segment_id:
            return seg["count"]
    return 0


def _parse_json(content: str) -> dict:
    cleaned = content.strip()
    # Strip markdown code fences if model ignores the no-fence instruction
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())
