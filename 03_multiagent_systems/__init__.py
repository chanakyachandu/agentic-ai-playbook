"""
Multi-Agent Systems: Hierarchical Swarms, Supervisor Orchestrator, Deep Research & Fact Verification.
100% Function-Based Architecture: Zero classes, pure composable Python functions.
"""

from .retrieval_worker import (
    run_retrieval_worker,
    search_policies,
    extract_policy_citations,
    deep_research_policy,
)
from .tool_worker import (
    run_tool_worker,
    fetch_order,
    calculate_date_metrics,
    calculate_financials,
)
from .compliance_worker import (
    run_compliance_worker,
    audit_invariants,
    enforce_financial_safety_gate,
)
from .supervisor_agent import (
    run_supervisor,
    decompose_inquiry,
    synthesize_resolution,
)
from .handoffs import (
    create_swarm_context,
    handoff,
    transfer_to_researcher,
    transfer_to_accountant,
    transfer_to_auditor,
    transfer_to_supervisor,
)
from .pipeline import run_multiagent_pipeline

__all__ = [
    "run_multiagent_pipeline",
    "run_supervisor",
    "run_retrieval_worker",
    "search_policies",
    "extract_policy_citations",
    "deep_research_policy",
    "run_tool_worker",
    "fetch_order",
    "calculate_date_metrics",
    "calculate_financials",
    "run_compliance_worker",
    "audit_invariants",
    "enforce_financial_safety_gate",
    "decompose_inquiry",
    "synthesize_resolution",
    "create_swarm_context",
    "handoff",
    "transfer_to_researcher",
    "transfer_to_accountant",
    "transfer_to_auditor",
    "transfer_to_supervisor",
]
