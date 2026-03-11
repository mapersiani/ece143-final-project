import logging
import pandas as pd
from typing import TypedDict, Callable
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
import os

from app.agents.analyst import run_analyst
from app.agents.strategist import run_strategist
from app.agents.critic import run_critic
from app.agents.executor import run_executor

logger = logging.getLogger(__name__)

MAX_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", 5))
CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", 7))


class DebateState(TypedDict):
    df: pd.DataFrame
    db: Session
    job_id: str
    analyst_report: dict
    proposals: list[dict]
    critiques: list[dict]
    round: int
    approved_plan: dict | None
    execution_package: dict | None
    status: str
    progress_cb: Callable | None  # optional progress callback for live DB updates


# ── Nodes ────────────────────────────────────────────────────────────────────

def analyst_node(state: DebateState) -> DebateState:
    logger.info(f"[{state['job_id']}] Analyst running")
    report = run_analyst(state["df"], state["db"])
    logger.info(f"[{state['job_id']}] Analyst done — {report.get('total_at_risk', 0)} at-risk customers, {len(report.get('segments', []))} segments")
    if state.get("progress_cb"):
        state["progress_cb"]("debating")
    return {**state, "analyst_report": report, "status": "debating"}


def strategist_node(state: DebateState) -> DebateState:
    round_num = state["round"] + 1
    logger.info(f"[{state['job_id']}] Strategist proposing (round {round_num})")
    proposals = run_strategist(state["analyst_report"], state["critiques"])
    logger.info(f"[{state['job_id']}] Strategist done (round {round_num})")
    return {**state, "proposals": state["proposals"] + [proposals], "round": round_num}


def critic_node(state: DebateState) -> DebateState:
    logger.info(f"[{state['job_id']}] Critic evaluating (round {state['round']})")
    latest_proposals = state["proposals"][-1]
    critique = run_critic(latest_proposals, state["analyst_report"], state["db"])
    rating = critique.get("overall_rating", "?")
    decision = critique.get("decision", "?")
    logger.info(f"[{state['job_id']}] Critic done — rating: {rating}/10, decision: {decision}")
    return {**state, "critiques": state["critiques"] + [critique]}


def executor_node(state: DebateState) -> DebateState:
    logger.info(f"[{state['job_id']}] Executor generating action plan")
    approved = state["proposals"][-1]
    pkg = run_executor(approved, state["analyst_report"], state["db"], state["job_id"])
    logger.info(f"[{state['job_id']}] Executor done — {len(pkg.get('campaigns', []))} campaigns created")
    if state.get("progress_cb"):
        state["progress_cb"]("approved")
    return {**state, "approved_plan": approved, "execution_package": pkg, "status": "approved"}


def escalate_node(state: DebateState) -> DebateState:
    logger.info(f"[{state['job_id']}] Max rounds reached — escalating to human review")
    if state.get("progress_cb"):
        state["progress_cb"]("escalated")
    return {**state, "status": "escalated"}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_critic(state: DebateState) -> str:
    latest_critique = state["critiques"][-1]
    overall_rating = latest_critique.get("overall_rating", 0)
    decision = latest_critique.get("decision", "revise")

    if decision == "approved" or overall_rating >= CONSENSUS_THRESHOLD:
        return "approved"
    if state["round"] >= MAX_ROUNDS:
        return "escalate"
    return "revise"


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(DebateState)

    g.add_node("analyst", analyst_node)
    g.add_node("strategist", strategist_node)
    g.add_node("critic", critic_node)
    g.add_node("executor", executor_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("analyst")
    g.add_edge("analyst", "strategist")
    g.add_edge("strategist", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"approved": "executor", "revise": "strategist", "escalate": "escalate"},
    )
    g.add_edge("executor", END)
    g.add_edge("escalate", END)

    return g.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(df: pd.DataFrame, db: Session, job_id: str, _progress_cb: Callable | None = None) -> dict:
    graph = get_graph()
    initial_state = DebateState(
        df=df,
        db=db,
        job_id=job_id,
        analyst_report={},
        proposals=[],
        critiques=[],
        round=0,
        approved_plan=None,
        execution_package=None,
        status="processing",
        progress_cb=_progress_cb,
    )
    final_state = graph.invoke(initial_state)
    return {
        "job_id": job_id,
        "status": final_state["status"],
        "analyst_report": final_state["analyst_report"],
        "debate_rounds": final_state["round"],
        "approved_plan": final_state.get("approved_plan"),
        "execution_package": final_state.get("execution_package"),
        "debate_log": {
            "proposals": final_state["proposals"],
            "critiques": final_state["critiques"],
        },
    }
