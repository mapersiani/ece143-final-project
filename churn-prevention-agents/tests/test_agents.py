import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# ── Analyst tests ─────────────────────────────────────────────────────────────

def _mock_df():
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "MonthlyCharges": [80.0, 50.0, 120.0],
        "tenure": [12, 2, 24],
        "TotalCharges": [960.0, 100.0, 2880.0],
    })


@patch("app.agents.analyst.predict_churn")
@patch("app.agents.analyst.get_top_churn_drivers")
@patch("app.agents.analyst.segment_customers")
@patch("app.agents.analyst.get_recent_actions")
@patch("app.agents.analyst.get_segment_profile")
def test_analyst_returns_report(mock_profile, mock_actions, mock_segments, mock_drivers, mock_predict):
    mock_predict.return_value = _mock_df().assign(
        churn_probability=[0.8, 0.9, 0.3], churn_risk=["high", "high", "low"]
    )
    mock_drivers.return_value = ["MonthlyCharges", "tenure"]
    mock_segments.return_value = {"driver_MonthlyCharges": [0, 1]}
    mock_actions.return_value = []
    mock_profile.return_value = None

    from app.agents.analyst import run_analyst
    db = MagicMock()
    report = run_analyst(_mock_df(), db)

    assert "segments" in report
    assert report["total_at_risk"] >= 0


# ── Graph routing tests ───────────────────────────────────────────────────────

def test_route_approved_on_high_rating():
    from app.agents.graph import route_after_critic
    state = {
        "critiques": [{"overall_rating": 8, "decision": "approved"}],
        "round": 2,
    }
    assert route_after_critic(state) == "approved"


def test_route_escalate_on_max_rounds():
    from app.agents.graph import route_after_critic, MAX_ROUNDS
    state = {
        "critiques": [{"overall_rating": 4, "decision": "revise"}],
        "round": MAX_ROUNDS,
    }
    assert route_after_critic(state) == "escalate"


def test_route_revise_on_low_rating():
    from app.agents.graph import route_after_critic
    state = {
        "critiques": [{"overall_rating": 4, "decision": "revise"}],
        "round": 1,
    }
    assert route_after_critic(state) == "revise"


# ── JSON parsing tests ────────────────────────────────────────────────────────

def test_parse_json_with_markdown_fence():
    from app.agents.strategist import _parse_json
    raw = '```json\n{"proposals": []}\n```'
    result = _parse_json(raw)
    assert result == {"proposals": []}


def test_parse_json_clean():
    from app.agents.critic import _parse_json
    raw = '{"overall_rating": 8, "decision": "approved"}'
    result = _parse_json(raw)
    assert result["decision"] == "approved"
