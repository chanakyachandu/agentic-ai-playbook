"""
Unit Tests for Module 2: Autonomous Agentic Patterns (100% Function-Based).
Verifies: Tool Execution, ReAct Loop, HITL Gate, Plan-and-Solve, and Self-Reflection.
"""

import pytest
import os
import sys
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions dynamically
tools_mod = importlib.import_module("02_agentic_patterns.tools")
react_mod = importlib.import_module("02_agentic_patterns.react_agent")
plan_mod = importlib.import_module("02_agentic_patterns.plan_and_solve")

lookup_order = tools_mod.lookup_order
check_return_policy = tools_mod.check_return_policy
calculate_refund = tools_mod.calculate_refund
process_refund = tools_mod.process_refund
execute_tool = tools_mod.execute_tool

run_react_agent = react_mod.run_react_agent
plan_tasks = plan_mod.plan_tasks
execute_plan = plan_mod.execute_plan
reflect_and_correct = plan_mod.reflect_and_correct


def test_lookup_order_found():
    order = lookup_order("ORD-2026-1001", db_path="data/ecommerce/orders_database.json")
    assert order["status"] == "FOUND"
    assert order["order_id"] == "ORD-2026-1001"
    assert "Apple MacBook Pro" in order["item_name"]
    assert order["unit_price_usd"] == 3499.00


def test_lookup_order_not_found():
    order = lookup_order("ORD-NON-EXISTENT", db_path="data/ecommerce/orders_database.json")
    assert order["status"] == "NOT_FOUND"


def test_check_return_policy():
    laptop_policy = check_return_policy("Laptops")
    assert laptop_policy["return_window_days"] == 14
    assert laptop_policy["max_runtime_hours"] == 10.0
    assert laptop_policy["restocking_fee_pct"] == 0.15

    general_policy = check_return_policy("Apparel")
    assert general_policy["return_window_days"] == 30
    assert general_policy["max_runtime_hours"] is None


def test_calculate_refund():
    calc = calculate_refund(1000.0, restocking_fee_pct=0.15)
    assert calc["original_amount_usd"] == 1000.0
    assert calc["restocking_fee_usd"] == 150.0
    assert calc["net_refund_usd"] == 850.0


def test_hitl_auto_approve():
    # <= $500 should auto approve
    res = process_refund("ORD-9999", 250.0, reason="Defective item", auto_approve_limit=500.0)
    assert res["status"] == "APPROVED"
    assert "confirmation_code" in res


def test_hitl_approval_required():
    # > $500 should trigger Human-in-the-Loop gate
    res = process_refund("ORD-9999", 1500.0, reason="High value return", auto_approve_limit=500.0)
    assert res["status"] == "REQUIRES_HUMAN_APPROVAL"
    assert "hitl_trigger" in res


def test_run_react_agent():
    query = "Customer wants to return order ORD-2026-1001. Please inspect order and calculate refund."
    result = run_react_agent(query, max_steps=5)

    assert result["query"] == query
    assert len(result["steps"]) >= 3
    assert result["hitl_triggered"] is True  # $3499 MacBook exceeds $500 limit
    assert "Final Answer" not in result["final_answer"] or len(result["final_answer"]) > 20
    assert "HUMAN APPROVAL REQUIRED" in result["final_answer"]


def test_plan_and_solve_and_reflect():
    goal = "Process return request for order ORD-2026-1001."
    plan = plan_tasks(goal)
    assert len(plan) == 4

    execution = execute_plan(goal)
    assert execution["order_id"] == "ORD-2026-1001"
    assert len(execution["step_results"]) == 4

    reflection = reflect_and_correct(goal, execution)
    assert "is_valid" in reflection
    assert len(reflection["critique_points"]) > 0
