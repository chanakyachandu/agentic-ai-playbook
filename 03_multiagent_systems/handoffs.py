"""
Autonomous Swarm Handoffs & State Passing.
100% Function-Based Architecture: Zero classes, pure composable Python functions.

Enables peer-to-peer and supervisor-worker task transitions with shared context.
"""

from typing import Dict, Any, Optional, Callable


def create_swarm_context(initial_query: str, order_id: Optional[str] = None) -> Dict[str, Any]:
    """Initializes the shared swarm context dictionary across all agents."""
    return {
        "query": initial_query,
        "order_id": order_id,
        "current_agent": "Supervisor",
        "handoff_history": ["Supervisor"],
        "context_variables": {},
        "shared_artifacts": {
            "retrieval_findings": None,
            "tool_findings": None,
            "compliance_findings": None,
        },
        "is_complete": False,
    }


def handoff(
    context: Dict[str, Any],
    target_agent: str,
    reason: str,
    payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes a dynamic peer-to-peer or supervisor-to-worker handoff.
    Updates the active agent and records the transition audit trail.
    """
    prev_agent = context.get("current_agent", "Supervisor")
    context["current_agent"] = target_agent
    context["handoff_history"].append(target_agent)

    if payload:
        context["context_variables"].update(payload)

    transition_record = {
        "from": prev_agent,
        "to": target_agent,
        "reason": reason,
    }
    if "transitions" not in context:
        context["transitions"] = []
    context["transitions"].append(transition_record)

    return context


def transfer_to_researcher(context: Dict[str, Any], topic: str) -> Dict[str, Any]:
    """Handoff trigger to dispatch the Retrieval Worker (Researcher)."""
    return handoff(
        context,
        target_agent="Researcher",
        reason=f"Deep research needed on policy topic: {topic}",
        payload={"research_topic": topic}
    )


def transfer_to_accountant(context: Dict[str, Any], order_id: str) -> Dict[str, Any]:
    """Handoff trigger to dispatch the Tool Worker (Accountant)."""
    return handoff(
        context,
        target_agent="Accountant",
        reason=f"Financial calculations & database lookup for order {order_id}",
        payload={"order_id": order_id}
    )


def transfer_to_auditor(context: Dict[str, Any], compliance_scope: str) -> Dict[str, Any]:
    """Handoff trigger to dispatch the Compliance Worker (Auditor)."""
    return handoff(
        context,
        target_agent="Auditor",
        reason=f"Audit invariants & enforce financial safety gates: {compliance_scope}",
        payload={"compliance_scope": compliance_scope}
    )


def transfer_to_supervisor(context: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """Handoff trigger to return control to the Supervisor for final synthesis."""
    return handoff(
        context,
        target_agent="Supervisor",
        reason=f"Worker task completed: {summary}",
        payload={"latest_summary": summary}
    )
