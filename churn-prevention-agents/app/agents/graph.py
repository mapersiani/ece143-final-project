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
    """
    Run the analyst step to compute churn risk segments and initialize the debate.

    :param state: Current debate state including input data and database session.
    :return: Updated state with an analyst report and status set to debating.
    """
    logger.info(f"[{state['job_id']}] Analyst running")
    report = run_analyst(state["df"], state["db"])
    logger.info(f"[{state['job_id']}] Analyst done — {report.get('total_at_risk', 0)} at-risk customers, {len(report.get('segments', []))} segments")
    if state.get("progress_cb"):
        state["progress_cb"]("debating")
    return {**state, "analyst_report": report, "status": "debating"}


def strategist_node(state: DebateState) -> DebateState:
    """
    Generate new strategist proposals and advance the debate round counter.

    :param state: Current debate state including analyst report and prior critiques.
    :return: Updated state with appended proposals and incremented round.
    """
    round_num = state["round"] + 1
    logger.info(f"[{state['job_id']}] Strategist proposing (round {round_num})")
    proposals = run_strategist(state["analyst_report"], state["critiques"])
    logger.info(f"[{state['job_id']}] Strategist done (round {round_num})")
    return {**state, "proposals": state["proposals"] + [proposals], "round": round_num}


def critic_node(state: DebateState) -> DebateState:
    """
    Evaluate the latest strategist proposals and append a critic response.

    :param state: Current debate state containing proposals and analyst context.
    :return: Updated state with an additional critique entry.
    """
    logger.info(f"[{state['job_id']}] Critic evaluating (round {state['round']})")
    latest_proposals = state["proposals"][-1]
    critique = run_critic(latest_proposals, state["analyst_report"], state["db"])
    rating = critique.get("overall_rating", "?")
    decision = critique.get("decision", "?")
    logger.info(f"[{state['job_id']}] Critic done — rating: {rating}/10, decision: {decision}")
    return {**state, "critiques": state["critiques"] + [critique]}


def executor_node(state: DebateState) -> DebateState:
    """
    Convert the final approved proposal into executable campaigns and update status.

    :param state: Current debate state including proposals and analyst report.
    :return: Updated state with an approved plan, execution package, and status.
    """
    logger.info(f"[{state['job_id']}] Executor generating action plan")
    approved = state["proposals"][-1]
    pkg = run_executor(approved, state["analyst_report"], state["db"], state["job_id"])
    logger.info(f"[{state['job_id']}] Executor done — {len(pkg.get('campaigns', []))} campaigns created")
    if state.get("progress_cb"):
        state["progress_cb"]("approved")
    return {**state, "approved_plan": approved, "execution_package": pkg, "status": "approved"}


def escalate_node(state: DebateState) -> DebateState:
    """
    Mark the debate as escalated to human review after exhausting allowed rounds.

    :param state: Current debate state when consensus was not reached.
    :return: Updated state with status set to escalated.
    """
    logger.info(f"[{state['job_id']}] Max rounds reached — escalating to human review")
    if state.get("progress_cb"):
        state["progress_cb"]("escalated")
    return {**state, "status": "escalated"}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_critic(state: DebateState) -> str:
    """
    Decide the next node after a critic pass based on rating, decision, and round count.

    :param state: Current debate state containing the latest critique and round info.
    :return: A routing label of 'approved', 'escalate', or 'revise'.
    """
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
    """
    Assemble and compile the debate graph connecting analyst, strategist, critic, and executor.

    :return: A compiled `StateGraph` ready to be invoked with a debate state.
    """
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
    """
    Return a singleton compiled debate graph, creating it on first access.

    :return: A compiled `StateGraph` instance used to run the debate pipeline.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(df: pd.DataFrame, db: Session, job_id: str, _progress_cb: Callable | None = None) -> dict:
    """
    Run the full debate pipeline from analyst through strategist, critic, and executor.

    :param df: Input customer DataFrame for analysis.
    :param db: Database session used by agents to read and persist context.
    :param job_id: Identifier for this pipeline execution for logging and persistence.
    :param _progress_cb: Optional callback invoked when major pipeline stages change.
    :return: A dictionary summarizing final status, reports, plans, and debate log.
    """
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
