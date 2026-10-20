"""
Supervisor Agent (The Boss).
100% Function-Based: Zero classes, pure composable Python functions.

Role: Coordinates the multi-agent swarm. Decomposes user inquiries,
delegates sub-tasks to specialists (Researcher, Accountant, Auditor),
and synthesizes findings into an auditable final resolution.
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    from .retrieval_worker import run_retrieval_worker
    from .tool_worker import run_tool_worker
    from .compliance_worker import run_compliance_worker
except ImportError:
    from retrieval_worker import run_retrieval_worker
    from tool_worker import run_tool_worker
    from compliance_worker import run_compliance_worker

load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    for cand in [
        os.path.join("..", ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"),
    ]:
        if os.path.exists(cand):
            load_dotenv(cand)
            break


def decompose_inquiry(query: str) -> Dict[str, Any]:
    """
    Extracts structured parameters from raw customer messages:
    - Order ID
    - Claimed runtime hours
    - Sanitary seal status
    - Dead pixel counts / Warranty claims
    - Defective flags
    """
    # 1. Order ID
    order_match = re.search(r"ORD-\d{4}-\d{4}", query.upper())
    order_id = order_match.group(0) if order_match else "ORD-2026-1001"

    # 2. Runtime hours
    runtime = None
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours|hrs|hr)", query, re.IGNORECASE)
    min_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|mins|min)", query, re.IGNORECASE)
    if hour_match:
        runtime = float(hour_match.group(1))
    elif min_match:
        runtime = round(float(min_match.group(1)) / 60.0, 2)

    # 3. Sanitary seal status
    broken_seal = bool(
        re.search(r"broke(?:n)?\s+(?:the\s+)?(?:plastic\s+)?sanitary\s+seal", query, re.IGNORECASE)
        or re.search(r"sanitary\s+seal\s+(?:is\s+)?broken", query, re.IGNORECASE)
    )

    # 4. Dead pixels (Warranty)
    dead_pixels = None
    pixel_match = re.search(r"(\d+)\s+(?:bright\s+)?dead\s+pixels?", query, re.IGNORECASE)
    if pixel_match:
        dead_pixels = int(pixel_match.group(1))

    # 5. Defective flag
    is_defective = bool(
        re.search(r"\b(defective|damaged|broken screen|dead pixels)\b", query, re.IGNORECASE)
    )

    return {
        "order_id": order_id,
        "claimed_runtime_hours": runtime,
        "sanitary_seal_broken": broken_seal,
        "dead_pixels_count": dead_pixels,
        "is_defective": is_defective,
    }


def synthesize_resolution(
    inquiry: str,
    tool_res: Dict[str, Any],
    retrieval_res: Dict[str, Any],
    compliance_res: Dict[str, Any],
    api_key: Optional[str] = None
) -> str:
    """
    Synthesizes the final resolution from worker outputs.
    Uses Gemini when configured, or generates a deterministic structured audit report.
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    decision = compliance_res.get("decision", "PENDING")
    is_hitl = compliance_res.get("hitl_triggered", False)
    approval_code = compliance_res.get("approval_code") or "N/A"
    order_id = tool_res.get("order_id", "N/A")
    item_name = tool_res.get("item_name", "Item")
    customer = tool_res.get("customer_name", "Valued Customer")
    fin = tool_res.get("financials", {})
    net_refund = fin.get("net_refund_usd", 0.0)
    fee_usd = fin.get("restocking_fee_usd", 0.0)
    fee_pct = int(fin.get("restocking_fee_pct", 0.0) * 100)

    # Gemini synthesis if live API available
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are the Supervisor AI Agent at NovaTech Electronics.\n"
                f"Synthesize the final customer response based on worker findings:\n"
                f"Customer Query: {inquiry}\n"
                f"Customer Name: {customer} | Order: {order_id} | Product: {item_name}\n"
                f"Compliance Decision: {decision}\n"
                f"HITL Approval Required: {is_hitl} (Approval Code: {approval_code})\n"
                f"Financials: Net Refund ${net_refund:.2f}, Fee ${fee_usd:.2f} ({fee_pct}%)\n"
                f"Violations: {compliance_res.get('violations', [])}\n"
                f"Approvals: {compliance_res.get('approvals', [])}\n"
                f"Citations: {[c['clause'] for c in retrieval_res.get('citations', [])]}\n\n"
                f"Requirements:\n"
                f"1. Be professional, clear, and empathetic.\n"
                f"2. Clearly state whether the request is approved, rejected, or paused for manager sign-off.\n"
                f"3. If HITL triggered, explicitly include 'HUMAN APPROVAL REQUIRED' and the approval hold code.\n"
                f"4. If rejected, clearly state the store policy rule that was violated."
            )
            resp = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            if resp and resp.text:
                return resp.text.strip()
        except Exception:
            pass

    # Deterministic Structured Output (Safe, Reliable Offline Fallback)
    lines = [
        f"DECISION: {decision}",
        f"Customer: {customer} | Order ID: {order_id} | Product: {item_name}",
    ]

    if decision == "REJECTED":
        lines.append("\nSTATUS: RETURN REQUEST REJECTED")
        lines.append("Policy Violations Found by Auditor:")
        for v in compliance_res.get("violations", []):
            lines.append(f" - {v}")
    elif decision == "REQUIRES_HUMAN_APPROVAL":
        lines.append("\nSTATUS: HUMAN APPROVAL REQUIRED")
        lines.append(f"Financial Safety Hold Code: {approval_code}")
        lines.append(f"Net Refund Amount: ${net_refund:.2f} (Restocking fee: ${fee_usd:.2f} @ {fee_pct}%)")
        lines.append(f"Auditor Note: {compliance_res.get('gate_message', '')}")
    else:
        lines.append("\nSTATUS: APPROVED")
        lines.append(f"Approval Code: {approval_code}")
        lines.append(f"Net Refund Amount: ${net_refund:.2f} (Restocking fee: ${fee_usd:.2f} @ {fee_pct}%)")
        for a in compliance_res.get("approvals", []):
            lines.append(f"Note: {a}")

    citations = retrieval_res.get("citations", [])
    if citations:
        lines.append("\nVerified Policy Citations:")
        for c in citations[:2]:
            lines.append(f" - [{c['source']}] {c['clause']}")

    return "\n".join(lines)


def run_supervisor(
    query: str,
    api_key: Optional[str] = None,
    data_dir: str = "data/ecommerce",
    auto_approve_limit: float = 500.0,
) -> Dict[str, Any]:
    """
    Main entry point for the Multi-Agent Supervisor Swarm.
    Coordinates Researcher, Accountant, and Auditor specialists.
    """
    # Step 1: Decompose query
    decomp = decompose_inquiry(query)

    # Step 2: Delegate to Tool Worker (Accountant)
    tool_res = run_tool_worker(
        order_id=decomp["order_id"],
        claimed_runtime_hours=decomp["claimed_runtime_hours"],
        is_defective=decomp["is_defective"],
    )

    category = tool_res.get("category", "General") if tool_res.get("status") == "FOUND" else None

    # Step 3: Delegate to Retrieval Worker (Researcher)
    retrieval_res = run_retrieval_worker(
        query=query,
        category=category,
        data_dir=data_dir,
    )

    # Step 4: Delegate to Compliance Worker (Auditor)
    compliance_res = run_compliance_worker(
        tool_result=tool_res,
        retrieval_result=retrieval_res,
        claimed_runtime_hours=decomp["claimed_runtime_hours"],
        sanitary_seal_broken=decomp["sanitary_seal_broken"],
        dead_pixels_count=decomp["dead_pixels_count"],
        auto_approve_limit=auto_approve_limit,
    )

    # Step 5: Synthesize Final Resolution
    resolution = synthesize_resolution(
        inquiry=query,
        tool_res=tool_res,
        retrieval_res=retrieval_res,
        compliance_res=compliance_res,
        api_key=api_key,
    )

    return {
        "query": query,
        "decomposition": decomp,
        "worker_outputs": {
            "retrieval_worker": retrieval_res,
            "tool_worker": tool_res,
            "compliance_worker": compliance_res,
        },
        "decision": compliance_res.get("decision", "UNKNOWN"),
        "hitl_triggered": compliance_res.get("hitl_triggered", False),
        "final_resolution": resolution,
    }


if __name__ == "__main__":
    test_q = (
        "Hi, I bought a MacBook Pro (order ORD-2026-1001). I used it for about 6 hours. "
        "I want to return it for a refund. Am I eligible and what is the fee?"
    )
    result = run_supervisor(test_q)
    print("=== SUPERVISOR SWARM RESULT ===")
    print(f"Decision: {result['decision']}")
    print(f"HITL Triggered: {result['hitl_triggered']}")
    print("\nResolution:\n" + result["final_resolution"])
