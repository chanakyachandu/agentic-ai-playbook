"""
Compliance Worker (The Auditor).
100% Function-Based: Zero classes, pure composable Python functions.

Role: Audits invariants, policy violations (sanitary seals, runtime caps, return windows),
and enforces the enterprise Human-in-the-Loop (HITL) financial safety gate for payments > $500.
"""

from typing import Dict, Any, List, Optional
import uuid


def audit_invariants(
    category: str,
    days_elapsed: int,
    runtime_hours: float,
    vip_tier: str = "Silver",
    sanitary_seal_broken: bool = False,
    dead_pixels_count: Optional[int] = None,
    factory_seal_intact: bool = False,
) -> Dict[str, Any]:
    """
    Audits policy compliance across categories and invariants.
    Returns whether the request is eligible or rejected, and the rationale.
    """
    violations: List[str] = []
    approvals: List[str] = []
    cat_lower = category.lower()

    # Rule 1: Section 1.4 - Earbuds sanitary seal
    if "earbud" in cat_lower or "in-ear" in cat_lower or "audio" in cat_lower:
        if sanitary_seal_broken:
            violations.append(
                "Section 1.4 Non-Returnable Item Violation: In-ear headphones and earbuds cannot be returned once sanitary plastic seal is broken."
            )

    # Rule 2: Section 3.2 - Display Warranty (Dead Pixels)
    if dead_pixels_count is not None and dead_pixels_count > 3:
        approvals.append(
            f"Section 3.2 Warranty Replacement Approved: Display defect with {dead_pixels_count} dead pixels exceeds threshold (> 3 dead pixels). Eligible for advance RMA exchange."
        )

    # Rule 3: Section 1.2 - Electronics Return Window & Runtime Hour Limits
    is_electronics = any(e in cat_lower for e in ["laptop", "camera", "drone", "smartphone", "computer"])
    if is_electronics:
        max_allowed_days = 14
        # Even VIP tiers are capped at 14 days for electronics runtime
        if days_elapsed > max_allowed_days:
            violations.append(
                f"Section 1.2 Electronics Window Violation: Electronics are restricted to a 14-day return window. Request submitted at {days_elapsed} days."
            )
        if runtime_hours > 10.0:
            violations.append(
                f"Section 1.2 Excessive Runtime Violation: Device usage ({runtime_hours} hrs) exceeds the strict 10-hour maximum runtime limit."
            )
    else:
        # Standard items: Silver 30 days, Gold 45 days, Platinum 60 days
        tier_limits = {"silver": 30, "gold": 45, "platinum": 60}
        allowed = tier_limits.get(vip_tier.lower(), 30)
        if days_elapsed > allowed:
            violations.append(
                f"Section 1.1 Return Window Violation: Return window of {allowed} days for {vip_tier} tier exceeded ({days_elapsed} days elapsed)."
            )

    is_eligible = len(violations) == 0

    return {
        "is_eligible": is_eligible,
        "violations": violations,
        "approvals": approvals,
    }


def enforce_financial_safety_gate(
    net_refund_usd: float,
    auto_approve_limit: float = 500.0,
) -> Dict[str, Any]:
    """
    Enforces Human-in-the-Loop financial safety limit.
    - Payments <= $500: Auto-approved immediately.
    - Payments > $500: Paused for human manager authorization.
    """
    if net_refund_usd <= auto_approve_limit:
        return {
            "status": "APPROVED",
            "hitl_triggered": False,
            "message": f"Refund of ${net_refund_usd:.2f} auto-approved under ${auto_approve_limit:.2f} threshold.",
            "approval_code": f"AUTO-{uuid.uuid4().hex[:8].upper()}",
        }
    else:
        return {
            "status": "REQUIRES_HUMAN_APPROVAL",
            "hitl_triggered": True,
            "message": f"Refund of ${net_refund_usd:.2f} exceeds financial safety limit (${auto_approve_limit:.2f}). Paused for manager sign-off.",
            "approval_code": f"HOLD-MGR-{uuid.uuid4().hex[:8].upper()}",
        }


def run_compliance_worker(
    tool_result: Dict[str, Any],
    retrieval_result: Optional[Dict[str, Any]] = None,
    claimed_runtime_hours: Optional[float] = None,
    sanitary_seal_broken: bool = False,
    dead_pixels_count: Optional[int] = None,
    auto_approve_limit: float = 500.0,
) -> Dict[str, Any]:
    """
    Main entry point for the Compliance Worker (Auditor).
    Evaluates invariants from the tool findings and enforces safety gates.
    """
    category = tool_result.get("category", "General")
    days_elapsed = tool_result.get("date_metrics", {}).get("days_elapsed", 0)
    runtime = (
        claimed_runtime_hours
        if claimed_runtime_hours is not None
        else tool_result.get("runtime_hours", 0.0)
    )
    vip_tier = tool_result.get("vip_tier", "Silver")
    factory_seal_intact = tool_result.get("factory_seal_intact", False)
    net_refund = tool_result.get("financials", {}).get("net_refund_usd", 0.0)

    # 1. Invariant audit
    audit = audit_invariants(
        category=category,
        days_elapsed=days_elapsed,
        runtime_hours=runtime,
        vip_tier=vip_tier,
        sanitary_seal_broken=sanitary_seal_broken,
        dead_pixels_count=dead_pixels_count,
        factory_seal_intact=factory_seal_intact,
    )

    # 2. Financial Gate
    if not audit["is_eligible"]:
        gate = {
            "status": "REJECTED",
            "hitl_triggered": False,
            "message": f"Return rejected due to {len(audit['violations'])} policy violation(s).",
            "approval_code": None,
        }
    else:
        gate = enforce_financial_safety_gate(net_refund, auto_approve_limit=auto_approve_limit)

    return {
        "worker": "Compliance Worker (Auditor)",
        "decision": gate["status"],
        "hitl_triggered": gate["hitl_triggered"],
        "is_eligible": audit["is_eligible"],
        "violations": audit["violations"],
        "approvals": audit["approvals"],
        "gate_message": gate["message"],
        "approval_code": gate["approval_code"],
    }


if __name__ == "__main__":
    mock_tool = {
        "category": "Laptops",
        "date_metrics": {"days_elapsed": 8},
        "runtime_hours": 6.5,
        "vip_tier": "Silver",
        "factory_seal_intact": False,
        "financials": {"net_refund_usd": 2974.15},
    }
    sample = run_compliance_worker(mock_tool)
    print(f"Worker: {sample['worker']}")
    print(f"Decision: {sample['decision']}")
    print(f"HITL Triggered: {sample['hitl_triggered']}")
    print(f"Gate Message: {sample['gate_message']}")
