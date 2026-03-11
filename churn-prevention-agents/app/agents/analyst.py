import pandas as pd
from sqlalchemy.orm import Session
from app.ml.model import predict_churn, get_top_churn_drivers, segment_customers
from app.db.memory import get_recent_actions, get_segment_profile

CHURN_THRESHOLD = 0.6


def run_analyst(df: pd.DataFrame, db: Session) -> dict:
    # predict_churn returns mock data regardless of input df
    predictions = predict_churn(df)
    at_risk = predictions[predictions["churn_probability"] >= CHURN_THRESHOLD].copy()
    at_risk = at_risk.reset_index(drop=True)

    if at_risk.empty:
        return {"segments": [], "total_at_risk": 0, "global_top_drivers": []}

    global_drivers = get_top_churn_drivers(at_risk)
    raw_segments = segment_customers(at_risk)

    segments = []
    for segment_id, customer_indices in raw_segments.items():
        # customer_indices may be positional ints or empty after filtering
        valid_indices = [i for i in customer_indices if i in at_risk.index]
        seg_df = at_risk.loc[valid_indices] if valid_indices else pd.DataFrame()
        if seg_df.empty:
            continue

        avg_churn_prob = float(seg_df["churn_probability"].mean())
        count = len(seg_df)

        # Estimate CLV proxy from tenure + monthly charges if available
        avg_clv = _estimate_clv(seg_df)

        # Pull memory context for this segment
        past_actions = get_recent_actions(db, [segment_id])
        profile = get_segment_profile(db, segment_id)

        memory_hint = _build_memory_hint(past_actions, profile)

        segments.append({
            "segment_id": segment_id,
            "count": count,
            "avg_churn_probability": round(avg_churn_prob, 3),
            "avg_clv_estimate": round(avg_clv, 2),
            "priority_score": round(avg_churn_prob * avg_clv, 2),
            "root_cause": segment_id.replace("driver_", "").replace("_", " "),
            "memory_hint": memory_hint,
        })

    # Sort by priority score descending
    segments.sort(key=lambda x: x["priority_score"], reverse=True)

    return {
        "segments": segments,
        "total_at_risk": len(at_risk),
        "global_top_drivers": global_drivers,
    }


def _estimate_clv(df: pd.DataFrame) -> float:
    # Use monthly charges * tenure as CLV proxy if columns exist
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        return float((df["MonthlyCharges"] * df["tenure"]).mean())
    if "MonthlyCharges" in df.columns:
        return float(df["MonthlyCharges"].mean() * 12)
    return 500.0  # fallback default


def _build_memory_hint(past_actions: list[dict], profile: dict | None) -> str:
    hints = []
    if profile and profile.get("avg_response_rate"):
        hints.append(f"Historical avg response rate: {profile['avg_response_rate']:.0%}")
    if profile and profile.get("best_channel"):
        hints.append(f"Best channel: {profile['best_channel']}")
    if past_actions:
        last = past_actions[0]
        outcome = last.get("outcome", {})
        if outcome:
            hints.append(
                f"Last action '{last['action_type']}' had {outcome.get('retention_rate', 'N/A')} retention rate"
            )
    return "; ".join(hints) if hints else "No historical data for this segment yet"
