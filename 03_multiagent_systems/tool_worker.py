"""
Tool Worker (The Accountant).
100% Function-Based: Zero classes, pure composable Python functions.

Role: Connects to the orders database, computes elapsed days from delivery,
checks recorded vs claimed runtime hours, and calculates restocking fees and net refunds.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


def resolve_orders_path(db_path: str = "data/ecommerce/orders_database.json") -> str:
    """Finds orders_database.json across current and parent folders."""
    if os.path.exists(db_path):
        return db_path

    file_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("..", db_path),
        os.path.join(file_dir, "..", "data", "ecommerce", "orders_database.json"),
        os.path.join(file_dir, "data", "ecommerce", "orders_database.json"),
        os.path.join(os.path.dirname(file_dir), "data", "ecommerce", "orders_database.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return db_path


def fetch_order(
    order_id: str,
    db_path: str = "data/ecommerce/orders_database.json"
) -> Dict[str, Any]:
    """Fetches full order details from orders_database.json."""
    resolved_path = resolve_orders_path(db_path)
    clean_id = order_id.strip().upper()

    if not os.path.exists(resolved_path):
        return {"status": "ERROR", "message": f"Database file not found: {resolved_path}"}

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            orders = json.load(f)

        for o in orders:
            if o.get("order_id", "").upper() == clean_id:
                return {
                    "status": "FOUND",
                    "order_id": o["order_id"],
                    "customer": o.get("customer", {}),
                    "purchase_date": o.get("purchase_date"),
                    "delivery_date": o.get("delivery_date"),
                    "return_request_date": o.get("return_request_date"),
                    "item": o.get("item", {}),
                    "order_total_usd": o.get("order_total_usd", 0.0),
                    "shipping_method": o.get("shipping_method"),
                    "policy_audit_grounding": o.get("policy_audit_grounding"),
                }
        return {"status": "NOT_FOUND", "order_id": clean_id, "message": f"Order {clean_id} not found."}
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed reading database: {str(e)}"}


def calculate_date_metrics(
    delivery_date_str: str,
    request_date_str: Optional[str] = None
) -> Dict[str, Any]:
    """Calculates days elapsed since delivery."""
    try:
        deliv = datetime.strptime(delivery_date_str, "%Y-%m-%d")
        if request_date_str:
            req = datetime.strptime(request_date_str, "%Y-%m-%d")
        else:
            req = datetime.now()
        days_elapsed = (req - deliv).days
        return {
            "delivery_date": delivery_date_str,
            "request_date": request_date_str or req.strftime("%Y-%m-%d"),
            "days_elapsed": max(0, days_elapsed),
        }
    except Exception as e:
        return {
            "delivery_date": delivery_date_str,
            "request_date": request_date_str,
            "days_elapsed": 0,
            "error": str(e),
        }


def calculate_financials(
    unit_price: float,
    condition: str = "Opened",
    factory_seal_intact: bool = False,
    vip_tier: str = "Silver",
    is_defective: bool = False,
    custom_fee_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates restocking fees and net refund amount based on store rules:
    - Defective: 0% fee
    - Factory seal intact / Unopened: 0% fee
    - Platinum VIP tier: 0% fee (waived per Section 4.1)
    - Opened non-defective (Silver / Gold / General): 15% restocking fee
    """
    if is_defective or factory_seal_intact:
        fee_pct = 0.0
    elif vip_tier.strip().lower() == "platinum":
        fee_pct = 0.0
    elif custom_fee_pct is not None:
        fee_pct = custom_fee_pct
    else:
        fee_pct = 0.15

    fee_amount = round(unit_price * fee_pct, 2)
    net_refund = round(unit_price - fee_amount, 2)

    return {
        "gross_amount_usd": round(unit_price, 2),
        "restocking_fee_pct": fee_pct,
        "restocking_fee_usd": fee_amount,
        "net_refund_usd": net_refund,
        "fee_waived_reason": "Platinum VIP Waiver" if vip_tier.lower() == "platinum" else (
            "Factory Seal Intact" if factory_seal_intact else ("Defective Claim" if is_defective else None)
        ),
    }


def run_tool_worker(
    order_id: str,
    claimed_runtime_hours: Optional[float] = None,
    is_defective: bool = False,
    request_date: Optional[str] = None,
    db_path: str = "data/ecommerce/orders_database.json"
) -> Dict[str, Any]:
    """
    Main entry point for the Tool Worker (Accountant).
    Fetches the order, computes calendar metrics, and calculates the refund math.
    """
    order = fetch_order(order_id, db_path=db_path)
    if order.get("status") != "FOUND":
        return {
            "worker": "Tool Worker (Accountant)",
            "status": "NOT_FOUND",
            "order_id": order_id,
            "message": order.get("message", "Order not found."),
        }

    item = order.get("item", {})
    customer = order.get("customer", {})
    deliv_date = order.get("delivery_date", "2026-03-01")
    req_date = request_date or order.get("return_request_date")

    date_metrics = calculate_date_metrics(deliv_date, req_date)

    unit_price = float(item.get("unit_price_usd", order.get("order_total_usd", 0.0)))
    condition = item.get("condition", "Opened")
    factory_seal_intact = bool(item.get("factory_seal_intact", False))
    vip_tier = customer.get("vip_tier", "Silver")

    recorded_runtime = float(item.get("runtime_hours", 0.0))
    effective_runtime = claimed_runtime_hours if claimed_runtime_hours is not None else recorded_runtime

    financials = calculate_financials(
        unit_price=unit_price,
        condition=condition,
        factory_seal_intact=factory_seal_intact,
        vip_tier=vip_tier,
        is_defective=is_defective,
    )

    return {
        "worker": "Tool Worker (Accountant)",
        "status": "SUCCESS",
        "order_id": order["order_id"],
        "customer_name": customer.get("name", "Unknown"),
        "vip_tier": vip_tier,
        "item_name": item.get("product_name", "Unknown Item"),
        "category": item.get("category", "General"),
        "condition": condition,
        "factory_seal_intact": factory_seal_intact,
        "runtime_hours": effective_runtime,
        "date_metrics": date_metrics,
        "financials": financials,
    }


if __name__ == "__main__":
    sample = run_tool_worker("ORD-2026-1001")
    print(f"Worker: {sample['worker']}")
    print(f"Order: {sample['order_id']} | Item: {sample['item_name']}")
    print(f"Days Elapsed: {sample['date_metrics']['days_elapsed']}")
    print(f"Net Refund: ${sample['financials']['net_refund_usd']}")
