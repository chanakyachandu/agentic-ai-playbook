"""
Plan-and-Solve Decomposition & Self-Reflection (Reflexion) Pattern.
100% Function-Based: No classes, pure Python functions for multi-step reasoning.
"""

import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

try:
    from .tools import get_tools_registry, execute_tool
except ImportError:
    from tools import get_tools_registry, execute_tool

load_dotenv()


def plan_tasks(goal: str) -> List[str]:
    """
    Decomposes a complex goal into a sequential plan of actionable sub-tasks.
    """
    order_match = re.search(r"ORD-\d{4}-\d{4}", goal.upper())
    order_id = order_match.group(0) if order_match else "Target Order"

    plan = [
        f"Step 1: Locate {order_id} in orders database and extract item category, delivery date, and purchase price.",
        f"Step 2: Check store return policy conditions for the item category (return window days & maximum runtime hours).",
        f"Step 3: Calculate net refund after applying restocking fee policies.",
        f"Step 4: Check Human-in-the-Loop (HITL) approval threshold ($500) and execute or hold refund.",
    ]
    return plan


def execute_plan(goal: str, tools_registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes the sequential plan step-by-step, aggregating findings into a unified result.
    """
    if tools_registry is None:
        tools_registry = get_tools_registry()

    plan = plan_tasks(goal)
    order_match = re.search(r"ORD-\d{4}-\d{4}", goal.upper())
    order_id = order_match.group(0) if order_match else "ORD-2026-1001"

    step_results = []

    # Step 1: Lookup
    res1 = execute_tool("lookup_order", {"order_id": order_id}, registry=tools_registry)
    step_results.append({"step": plan[0], "result": res1})

    category = res1.get("category", "Laptops")
    price = res1.get("unit_price_usd", 1000.0)

    # Step 2: Policy
    res2 = execute_tool("check_return_policy", {"category": category}, registry=tools_registry)
    step_results.append({"step": plan[1], "result": res2})

    # Step 3: Calculate
    fee_pct = res2.get("restocking_fee_pct", 0.15)
    res3 = execute_tool("calculate_refund", {"unit_price": price, "restocking_fee_pct": fee_pct}, registry=tools_registry)
    step_results.append({"step": plan[2], "result": res3})

    net_refund = res3.get("net_refund_usd", price)

    # Step 4: Process
    res4 = execute_tool("process_refund", {"order_id": order_id, "refund_amount": net_refund, "reason": "Plan-and-Solve Return Execution"}, registry=tools_registry)
    step_results.append({"step": plan[3], "result": res4})

    hitl = res4.get("status") == "REQUIRES_HUMAN_APPROVAL"

    return {
        "goal": goal,
        "plan": plan,
        "step_results": step_results,
        "order_id": order_id,
        "net_refund_usd": net_refund,
        "hitl_triggered": hitl,
        "status": "REQUIRES_APPROVAL" if hitl else "COMPLETED",
    }


def reflect_and_correct(goal: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflexion / Critic Function:
    Evaluates execution result against the original goal for completeness, policy compliance, and safety.
    """
    critique_points = []
    corrections = []

    # Check 1: Was order found?
    step1 = execution_result["step_results"][0]["result"]
    if step1.get("status") != "FOUND":
        critique_points.append("CRITIQUE: Order could not be validated in records.")
        corrections.append("CORRECTION: Prompt user to provide a valid ORD-XXXX format ID.")

    # Check 2: Were high-value electronics inspected for runtime hours?
    step2 = execution_result["step_results"][1]["result"]
    if step2.get("max_runtime_hours") is not None:
        runtime = step1.get("runtime_hours", 0.0)
        max_h = step2["max_runtime_hours"]
        if runtime > max_h:
            critique_points.append(f"CRITIQUE: Runtime hours ({runtime} hrs) exceed policy maximum ({max_h} hrs).")
            corrections.append("CORRECTION: Return should be rejected or flagged for exception override.")
        else:
            critique_points.append(f"PASS: Runtime hours ({runtime} hrs) within allowable limit ({max_h} hrs).")

    # Check 3: Human-in-the-Loop check
    if execution_result.get("hitl_triggered"):
        critique_points.append("PASS: High-value refund correctly quarantined for manager review.")
    else:
        critique_points.append("PASS: Standard refund automatically cleared.")

    is_valid = len(corrections) == 0

    return {
        "is_valid": is_valid,
        "critique_points": critique_points,
        "corrections": corrections,
        "final_recommendation": "Ready for dispatch to customer." if is_valid else "Hold dispatch: resolve corrections.",
    }
