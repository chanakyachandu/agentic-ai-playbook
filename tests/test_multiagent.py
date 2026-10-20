"""
Unit & Integration Tests for Module 3: Multi-Agent Systems (100% Function-Based).
Verifies: Retrieval Worker, Tool Worker, Compliance Worker, Supervisor Swarm, and Master Pipeline.
"""

import os
import sys
import importlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions dynamically to support all test runners
retrieval_mod = importlib.import_module("03_multiagent_systems.retrieval_worker")
tool_mod = importlib.import_module("03_multiagent_systems.tool_worker")
compliance_mod = importlib.import_module("03_multiagent_systems.compliance_worker")
supervisor_mod = importlib.import_module("03_multiagent_systems.supervisor_agent")
pipeline_mod = importlib.import_module("03_multiagent_systems.pipeline")

run_retrieval_worker = retrieval_mod.run_retrieval_worker
search_policies = retrieval_mod.search_policies
extract_policy_citations = retrieval_mod.extract_policy_citations

run_tool_worker = tool_mod.run_tool_worker
fetch_order = tool_mod.fetch_order
calculate_financials = tool_mod.calculate_financials

run_compliance_worker = compliance_mod.run_compliance_worker
audit_invariants = compliance_mod.audit_invariants
enforce_financial_safety_gate = compliance_mod.enforce_financial_safety_gate

run_supervisor = supervisor_mod.run_supervisor
decompose_inquiry = supervisor_mod.decompose_inquiry

run_multiagent_pipeline = pipeline_mod.run_multiagent_pipeline


def test_retrieval_worker_policy_search():
    """Researcher finds policy clauses and extracts verifiable citations."""
    res = run_retrieval_worker("What is the return window for laptops and restocking fee?", category="Laptops")
    assert res["status"] == "SUCCESS"
    assert res["document_count"] > 0
    assert len(res["citations"]) > 0
    assert any("2026_current_policy.txt" in c["source"] for c in res["citations"])


def test_tool_worker_order_lookup_and_math():
    """Accountant fetches order ORD-2026-1001, checks days and calculates 15% fee."""
    res = run_tool_worker("ORD-2026-1001", claimed_runtime_hours=6.5)
    assert res["status"] == "SUCCESS"
    assert res["order_id"] == "ORD-2026-1001"
    assert res["customer_name"] == "John Doe"
    assert res["vip_tier"] == "Silver"
    assert res["category"] == "Laptops"
    assert res["financials"]["gross_amount_usd"] == 3499.00
    assert res["financials"]["restocking_fee_usd"] == 524.85
    assert res["financials"]["net_refund_usd"] == 2974.15


def test_tool_worker_platinum_fee_waiver():
    """Accountant waives restocking fees for Platinum VIP member."""
    res = run_tool_worker("ORD-2026-1003")
    assert res["status"] == "SUCCESS"
    assert res["vip_tier"] == "Platinum"
    assert res["financials"]["restocking_fee_pct"] == 0.0
    assert res["financials"]["restocking_fee_usd"] == 0.0
    assert res["financials"]["fee_waived_reason"] is not None


def test_compliance_worker_earbuds_seal_violation():
    """Auditor strictly rejects returns on earbuds with broken sanitary seals."""
    tool_mock = {
        "category": "Audio / Earbuds",
        "date_metrics": {"days_elapsed": 3},
        "runtime_hours": 0.33,
        "vip_tier": "Silver",
        "factory_seal_intact": False,
        "financials": {"net_refund_usd": 250.0},
    }
    audit = run_compliance_worker(tool_mock, sanitary_seal_broken=True)
    assert audit["decision"] == "REJECTED"
    assert audit["is_eligible"] is False
    assert any("Section 1.4" in v for v in audit["violations"])
    assert audit["hitl_triggered"] is False


def test_compliance_worker_excessive_runtime():
    """Auditor rejects electronics with > 10 hours runtime."""
    tool_mock = {
        "category": "Digital Cameras",
        "date_metrics": {"days_elapsed": 12},
        "runtime_hours": 34.0,
        "vip_tier": "Gold",
        "factory_seal_intact": False,
        "financials": {"net_refund_usd": 2123.30},
    }
    audit = run_compliance_worker(tool_mock)
    assert audit["decision"] == "REJECTED"
    assert audit["is_eligible"] is False
    assert any("Section 1.2" in v for v in audit["violations"])


def test_compliance_worker_display_warranty():
    """Auditor approves advance RMA warranty replacement for > 3 dead pixels."""
    tool_mock = {
        "category": "Monitors & Displays",
        "date_metrics": {"days_elapsed": 20},
        "runtime_hours": 50.0,
        "vip_tier": "Silver",
        "factory_seal_intact": False,
        "financials": {"net_refund_usd": 899.0},
    }
    audit = run_compliance_worker(tool_mock, dead_pixels_count=5)
    assert audit["is_eligible"] is True
    assert len(audit["approvals"]) > 0
    assert any("Section 3.2" in a for a in audit["approvals"])


def test_compliance_worker_hitl_financial_gate():
    """Financial safety gate: auto-approves <= $500, flags > $500."""
    # Under $500: auto-approved
    auto_res = enforce_financial_safety_gate(250.0, auto_approve_limit=500.0)
    assert auto_res["status"] == "APPROVED"
    assert auto_res["hitl_triggered"] is False
    assert "AUTO-" in auto_res["approval_code"]

    # Over $500: pauses for human approval
    hold_res = enforce_financial_safety_gate(1200.0, auto_approve_limit=500.0)
    assert hold_res["status"] == "REQUIRES_HUMAN_APPROVAL"
    assert hold_res["hitl_triggered"] is True
    assert "HOLD-MGR-" in hold_res["approval_code"]


def test_supervisor_decomposition():
    """Supervisor parses raw customer inquiry into structured signals."""
    msg = (
        "Hi, I bought a MacBook Pro (order ORD-2026-1001). I used it for about 6 hours. "
        "I want to return it for a refund."
    )
    decomp = decompose_inquiry(msg)
    assert decomp["order_id"] == "ORD-2026-1001"
    assert decomp["claimed_runtime_hours"] == 6.0
    assert decomp["sanitary_seal_broken"] is False


def test_supervisor_end_to_end_hitl():
    """Supervisor coordinates swarm on $3499 MacBook return, triggering HITL gate."""
    msg = (
        "Hi, I bought order ORD-2026-1001. I used it for 6 hours. "
        "I want to return it for a refund to my credit card."
    )
    swarm = run_supervisor(msg)
    assert swarm["decision"] == "REQUIRES_HUMAN_APPROVAL"
    assert swarm["hitl_triggered"] is True
    assert "HUMAN APPROVAL REQUIRED" in swarm["final_resolution"]
    assert "worker_outputs" in swarm
    assert "retrieval_worker" in swarm["worker_outputs"]
    assert "tool_worker" in swarm["worker_outputs"]
    assert "compliance_worker" in swarm["worker_outputs"]


def test_supervisor_end_to_end_rejection():
    """Supervisor coordinates swarm on earbuds with broken seal, resulting in rejection."""
    msg = (
        "Order ORD-2026-1004. I opened the box and broke the plastic sanitary seal "
        "to test the earbuds. Can I get a refund?"
    )
    swarm = run_supervisor(msg)
    assert swarm["decision"] == "REJECTED"
    assert swarm["hitl_triggered"] is False
    assert "REJECTED" in swarm["final_resolution"]
    assert "Section 1.4" in swarm["final_resolution"]


def test_multiagent_pipeline_single_call():
    """Master single-call pipeline runs end-to-end and measures latency."""
    msg = "Customer requests return on order ORD-2026-1001. Runtime usage: 6.5 hours."
    res = run_multiagent_pipeline(msg)
    assert "query" in res
    assert "decision" in res
    assert "resolution" in res
    assert "latency_ms" in res
    assert res["latency_ms"] >= 0.0
    assert res["hitl_triggered"] is True


def test_deep_research_policy():
    """Deep Researcher runs iterative sub-queries and aggregates findings."""
    deep_res = retrieval_mod.deep_research_policy(
        topic="Laptop returns and warranty",
        category="Laptops",
        vip_tier="Silver"
    )
    assert deep_res["worker"] == "Retrieval Worker (Deep Researcher)"
    assert len(deep_res["sub_queries_executed"]) == 4
    assert deep_res["total_documents_analyzed"] > 0
    assert len(deep_res["verified_citations"]) > 0
    assert len(deep_res["key_findings"]) > 0


def test_swarm_handoffs():
    """Autonomous Swarm Handoff mechanism transitions between specialists."""
    handoffs_mod = importlib.import_module("03_multiagent_systems.handoffs")
    create_swarm_context = handoffs_mod.create_swarm_context
    transfer_to_accountant = handoffs_mod.transfer_to_accountant
    transfer_to_auditor = handoffs_mod.transfer_to_auditor
    transfer_to_supervisor = handoffs_mod.transfer_to_supervisor

    ctx = create_swarm_context("Process return for ORD-2026-1001", order_id="ORD-2026-1001")
    assert ctx["current_agent"] == "Supervisor"

    # Handoff to Accountant
    ctx = transfer_to_accountant(ctx, "ORD-2026-1001")
    assert ctx["current_agent"] == "Accountant"
    assert "Accountant" in ctx["handoff_history"]

    # Handoff to Auditor
    ctx = transfer_to_auditor(ctx, "financial and policy check")
    assert ctx["current_agent"] == "Auditor"

    # Return to Supervisor
    ctx = transfer_to_supervisor(ctx, "Audit complete with manager hold")
    assert ctx["current_agent"] == "Supervisor"
    assert len(ctx["transitions"]) == 3

