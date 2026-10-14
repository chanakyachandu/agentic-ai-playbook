"""
Enterprise Agent Tools for Customer Care & Order Management.
100% Function-Based: No classes, pure Python functions with Human-in-the-Loop (HITL) gate.
"""

import json
import os
from typing import Dict, Any, List, Optional


def lookup_order(order_id: str, db_path: str = "data/ecommerce/orders_database.json") -> Dict[str, Any]:
    """
    Looks up full order details by order_id (e.g. 'ORD-2026-1001').
    Returns customer info, item details, delivery date, and order total.
    """
    clean_id = order_id.strip().upper()
    if not os.path.exists(db_path):
        # Fallback path if running from subdirectories
        alt_path = os.path.join("..", db_path)
        if os.path.exists(alt_path):
            db_path = alt_path
        else:
            return {"error": f"Database file not found: {db_path}"}

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            orders = json.load(f)

        for order in orders:
            if order.get("order_id", "").upper() == clean_id:
                return {
                    "status": "FOUND",
                    "order_id": order["order_id"],
                    "customer_name": order["customer"]["name"],
                    "vip_tier": order["customer"]["vip_tier"],
                    "item_name": order["item"]["product_name"],
                    "category": order["item"]["category"],
                    "unit_price_usd": order["item"]["unit_price_usd"],
                    "condition": order["item"]["condition"],
                    "runtime_hours": order["item"].get("runtime_hours", 0.0),
                    "delivery_date": order["delivery_date"],
                    "order_total_usd": order["order_total_usd"],
                }
        return {"status": "NOT_FOUND", "message": f"Order ID {clean_id} does not exist in system."}
    except Exception as e:
        return {"error": f"Failed reading orders database: {str(e)}"}


def check_return_policy(category: str) -> Dict[str, Any]:
    """
    Checks store return policy limits for a given product category.
    Electronics, Laptops, Cameras, Drones have strict 14-day limits and <10 runtime hours.
    Standard items have a 30-day window.
    """
    cat_lower = category.strip().lower()
    high_value_categories = ["laptop", "laptops", "camera", "cameras", "digital cameras", "drone", "drones", "smartphone", "smartphones"]

    is_high_value = any(term in cat_lower for term in high_value_categories)

    if is_high_value:
        return {
            "category": category,
            "return_window_days": 14,
            "max_runtime_hours": 10.0,
            "restocking_fee_pct": 0.15,
            "requires_original_packaging": True,
            "policy_clause": "Section 1.2: High-value electronics limited to 14 days and <10 runtime hours.",
        }
    else:
        return {
            "category": category,
            "return_window_days": 30,
            "max_runtime_hours": None,
            "restocking_fee_pct": 0.0,
            "requires_original_packaging": True,
            "policy_clause": "Section 1.1: Standard items eligible for 30 calendar days full refund.",
        }


def calculate_refund(unit_price: float, restocking_fee_pct: float = 0.15, waive_fee: bool = False) -> Dict[str, Any]:
    """
    Calculates net refund amount after applicable restocking fees.
    """
    try:
        price = float(unit_price)
        pct = 0.0 if waive_fee else float(restocking_fee_pct)
        fee_amount = round(price * pct, 2)
        net_refund = round(price - fee_amount, 2)
        return {
            "original_amount_usd": price,
            "restocking_fee_pct": pct,
            "restocking_fee_usd": fee_amount,
            "net_refund_usd": net_refund,
            "fee_waived": waive_fee,
        }
    except Exception as e:
        return {"error": f"Invalid calculation parameters: {str(e)}"}


def process_refund(order_id: str, refund_amount: float, reason: str, auto_approve_limit: float = 500.0) -> Dict[str, Any]:
    """
    Executes refund with Human-in-the-Loop (HITL) safety constraint:
    - Refunds <= $500: Automatically approved and processed.
    - Refunds > $500: Paused with status 'REQUIRES_HUMAN_APPROVAL'.
    """
    try:
        amount = float(refund_amount)
        if amount <= 0:
            return {"status": "REJECTED", "message": "Refund amount must be positive."}

        if amount > auto_approve_limit:
            return {
                "status": "REQUIRES_HUMAN_APPROVAL",
                "order_id": order_id,
                "requested_amount_usd": amount,
                "reason": reason,
                "hitl_trigger": f"Amount ${amount:.2f} exceeds auto-approval threshold (${auto_approve_limit:.2f}).",
                "action_required": "Manager authorization required before funds are disbursed.",
            }
        else:
            return {
                "status": "APPROVED",
                "order_id": order_id,
                "refunded_amount_usd": amount,
                "reason": reason,
                "confirmation_code": f"REF-{order_id.replace('ORD-', '')}-AUTO",
                "message": f"Refund of ${amount:.2f} successfully processed back to original payment card.",
            }
    except Exception as e:
        return {"error": f"Failed processing refund: {str(e)}"}


def get_tools_registry() -> Dict[str, Any]:
    """Returns mapping of tool names to callable functions and descriptions."""
    return {
        "lookup_order": {
            "function": lookup_order,
            "description": "lookup_order(order_id: str) - Looks up customer, item, and delivery details for an order ID.",
        },
        "check_return_policy": {
            "function": check_return_policy,
            "description": "check_return_policy(category: str) - Checks return window and runtime hour restrictions for a product category.",
        },
        "calculate_refund": {
            "function": calculate_refund,
            "description": "calculate_refund(unit_price: float, restocking_fee_pct: float, waive_fee: bool) - Computes net refund after restocking fees.",
        },
        "process_refund": {
            "function": process_refund,
            "description": "process_refund(order_id: str, refund_amount: float, reason: str) - Submits refund. Amounts > $500 trigger Human-in-the-Loop approval.",
        },
    }


def execute_tool(tool_name: str, kwargs: Dict[str, Any], registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safely executes a registered tool function by name."""
    if registry is None:
        registry = get_tools_registry()

    clean_name = tool_name.strip()
    if clean_name not in registry:
        return {"error": f"Unknown tool '{clean_name}'. Available tools: {list(registry.keys())}"}

    func = registry[clean_name]["function"]
    try:
        return func(**kwargs)
    except TypeError as te:
        return {"error": f"Invalid arguments for tool '{clean_name}': {str(te)}"}
    except Exception as e:
        return {"error": f"Error running tool '{clean_name}': {str(e)}"}
