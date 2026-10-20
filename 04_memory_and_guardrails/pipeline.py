"""
Master Pipeline for Memory & Guardrails (Project 4).
100% Function-Based: Pure, composable Python functions with zero class boilerplate.

Workflow:
1. Session Memory: Loads session, adds message, generates rolling summary if needed, and extracts sliding window.
2. SQL AST Guardrail: Intercepts queries, blocks destructive statements (DROP/DELETE), and enforces LIMIT 50.
3. Schema & PII Guardrail: Verifies structured output fields and redacts sensitive PII (CC, SSN, email).
4. Execution & Audit: Records execution latency, memory state, and guardrail verdicts in a unified payload.
"""

import time
import sys
import os
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .memory_manager import (
        create_memory_session,
        add_message,
        get_sliding_window,
        summarize_history,
        build_memory_context,
    )
    from .sql_guardrails import validate_sql_query, enforce_query_limits
    from .schema_guardrail import validate_response_schema, sanitize_pii
except ImportError:
    from memory_manager import (
        create_memory_session,
        add_message,
        get_sliding_window,
        summarize_history,
        build_memory_context,
    )
    from sql_guardrails import validate_sql_query, enforce_query_limits
    from schema_guardrail import validate_response_schema, sanitize_pii


# Active in-memory session registry
SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(session_id: str, max_window: int = 4) -> Dict[str, Any]:
    """
    Retrieves an existing memory session from registry or creates a new one.
    """
    if session_id not in SESSION_REGISTRY:
        SESSION_REGISTRY[session_id] = create_memory_session(session_id=session_id, max_window=max_window)
    return SESSION_REGISTRY[session_id]


def run_memory_guardrails_pipeline(
    session_id: str,
    user_message: str,
    sql_query: Optional[str] = None,
    mock_agent_output: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-call entry point executing Memory management + Security Guardrails.
    """
    start_time = time.time()

    # Step 1: Manage Session Memory
    session = get_or_create_session(session_id)
    add_message(session, role="user", content=user_message)

    # Trigger rolling summary if turns exceed sliding window
    summarize_history(session, api_key=api_key)
    memory_context = build_memory_context(session)

    # Step 2: Audit SQL Query (if SQL execution requested)
    sql_audit = None
    guarded_sql = None
    sql_blocked = False

    if sql_query:
        validation = validate_sql_query(sql_query)
        if validation["is_safe"]:
            guarded_sql = enforce_query_limits(validation["sanitized_sql"], default_limit=50, max_limit=100)
            sql_audit = {
                "status": "APPROVED",
                "original_sql": sql_query,
                "guarded_sql": guarded_sql,
                "reason": "Passed AST safety verification; LIMIT clause enforced.",
            }
        else:
            sql_blocked = True
            sql_audit = {
                "status": "BLOCKED",
                "original_sql": sql_query,
                "guarded_sql": None,
                "reason": validation["reason"],
                "ast_type": validation["ast_type"],
            }

    # Step 3: Schema & PII Output Guardrail
    if sql_blocked:
        agent_reply_text = f"SECURITY ALERT: SQL query blocked by AST guardrail ({sql_audit['reason']})."
        output_schema_audit = {"is_valid": True, "errors": [], "sanitized_payload": {"status": "BLOCKED"}}
    else:
        # Default response payload if none provided
        payload = mock_agent_output or {
            "status": "SUCCESS",
            "message": f"Processed user request with {len(get_sliding_window(session))} active turns in memory.",
            "refund_amount": 150.00,
        }
        output_schema_audit = validate_response_schema(
            payload=payload,
            required_fields=["status", "message"],
            field_types={"status": str, "message": str},
        )
        agent_reply_text = sanitize_pii(payload.get("message", "Request completed."))

    # Step 4: Add assistant response to session memory
    add_message(session, role="assistant", content=agent_reply_text)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "session_id": session_id,
        "user_message": sanitize_pii(user_message),
        "memory_context": memory_context,
        "sliding_window_turns": len(get_sliding_window(session)),
        "rolling_summary": session.get("rolling_summary", ""),
        "sql_audit": sql_audit,
        "output_schema_audit": output_schema_audit,
        "final_response": agent_reply_text,
        "latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Testing Memory & Guardrails Pipeline ===")
    
    # 1. Normal safe query
    res1 = run_memory_guardrails_pipeline(
        session_id="USER-DEMO-01",
        user_message="Find my laptop orders. My email is john.doe@example.com",
        sql_query="SELECT * FROM orders WHERE category = 'Laptops'",
    )
    print(f"Turn 1 SQL Status: {res1['sql_audit']['status']}")
    print(f"Guarded SQL: {res1['sql_audit']['guarded_sql']}")
    print(f"Sanitized Message: {res1['user_message']}")

    # 2. Malicious injection attempt
    res2 = run_memory_guardrails_pipeline(
        session_id="USER-DEMO-01",
        user_message="Delete all records!",
        sql_query="SELECT 1; DROP TABLE orders;",
    )
    print(f"\nTurn 2 SQL Status: {res2['sql_audit']['status']}")
    print(f"Block Reason: {res2['sql_audit']['reason']}")
    print(f"Final Response: {res2['final_response']}")
