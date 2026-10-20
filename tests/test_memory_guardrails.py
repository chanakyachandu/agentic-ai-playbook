"""
Automated Pytest Suite for Project 4: Memory & Guardrails.
Verifies sliding-window memory, rolling summaries, SQL AST validation,
query limit caps, PII sanitization, and end-to-end guarded pipeline execution.
"""

import os
import sys
import importlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Dynamic imports to support numeric folder naming
mem_mod = importlib.import_module("04_memory_and_guardrails.memory_manager")
sql_mod = importlib.import_module("04_memory_and_guardrails.sql_guardrails")
schema_mod = importlib.import_module("04_memory_and_guardrails.schema_guardrail")
pipe_mod = importlib.import_module("04_memory_and_guardrails.pipeline")

create_memory_session = mem_mod.create_memory_session
add_message = mem_mod.add_message
get_sliding_window = mem_mod.get_sliding_window
summarize_history = mem_mod.summarize_history
build_memory_context = mem_mod.build_memory_context
clear_memory_session = mem_mod.clear_memory_session

validate_sql_query = sql_mod.validate_sql_query
enforce_query_limits = sql_mod.enforce_query_limits

sanitize_pii = schema_mod.sanitize_pii
validate_response_schema = schema_mod.validate_response_schema

run_memory_guardrails_pipeline = pipe_mod.run_memory_guardrails_pipeline


# =====================================================================
# 1. Memory Manager Tests
# =====================================================================

def test_create_and_add_memory():
    session = create_memory_session(session_id="TEST-01", max_window=3)
    assert session["session_id"] == "TEST-01"
    assert session["turn_count"] == 0

    add_message(session, role="user", content="Hello!")
    add_message(session, role="assistant", content="How can I help?")

    assert session["turn_count"] == 2
    assert len(session["messages"]) == 2
    assert session["messages"][0]["role"] == "user"


def test_sliding_window_retention():
    session = create_memory_session(session_id="TEST-02", max_window=3)
    for i in range(6):
        add_message(session, role="user", content=f"Message {i+1}")

    assert session["turn_count"] == 6
    window = get_sliding_window(session)
    assert len(window) == 3
    # Should only contain the latest 3 turns (4, 5, 6)
    assert window[0]["content"] == "Message 4"
    assert window[2]["content"] == "Message 6"


def test_rolling_summary_generation():
    session = create_memory_session(session_id="TEST-03", max_window=2)
    add_message(session, role="user", content="I ordered a MacBook Pro ORD-2026-1001.")
    add_message(session, role="assistant", content="I see your order. What is the issue?")
    add_message(session, role="user", content="The screen has a defect.")
    add_message(session, role="assistant", content="Let me check warranty policy.")

    # 4 turns total, max_window is 2, so turns 1 and 2 slide into summary
    summary = summarize_history(session)
    assert summary != ""
    assert "user" in summary.lower() or "ord-2026-1001" in summary.lower()
    assert session["rolling_summary"] != ""


def test_build_memory_context():
    session = create_memory_session(session_id="TEST-04", max_window=2)
    add_message(session, role="user", content="Where is my refund?")
    add_message(session, role="assistant", content="Checking your status now.")

    ctx = build_memory_context(session)
    assert "[ACTIVE CONVERSATION TURNS]" in ctx
    assert "Where is my refund?" in ctx


def test_clear_memory_session():
    session = create_memory_session(session_id="TEST-05", max_window=2)
    add_message(session, role="user", content="Hi")
    clear_memory_session(session)
    assert session["turn_count"] == 0
    assert len(session["messages"]) == 0
    assert session["rolling_summary"] == ""


# =====================================================================
# 2. SQL AST Guardrail Tests
# =====================================================================

def test_sql_ast_allows_safe_select():
    query = "SELECT order_id, product_name, total_usd FROM orders WHERE status = 'COMPLETED'"
    res = validate_sql_query(query)
    assert res["is_safe"] is True
    assert res["ast_type"] == "SELECT"
    assert res["reason"] is None


def test_sql_ast_blocks_destructive_operations():
    destructive_queries = [
        "DROP TABLE customers;",
        "DELETE FROM orders WHERE 1=1;",
        "TRUNCATE TABLE transactions;",
        "ALTER TABLE users ADD COLUMN password_hash TEXT;",
        "UPDATE orders SET total_usd = 0.0;",
        "INSERT INTO users VALUES ('hacker', 'admin');",
    ]
    for q in destructive_queries:
        res = validate_sql_query(q)
        assert res["is_safe"] is False, f"Expected query to be blocked: {q}"
        assert res["reason"] is not None


def test_sql_ast_blocks_multi_statement_injection():
    attack = "SELECT * FROM products; DROP TABLE products;"
    res = validate_sql_query(attack)
    assert res["is_safe"] is False
    assert res["ast_type"] == "MULTI_STATEMENT"
    assert "Multiple stacked" in res["reason"]


def test_sql_enforce_missing_limit():
    query = "SELECT * FROM orders WHERE category = 'Audio'"
    guarded = enforce_query_limits(query, default_limit=50, max_limit=100)
    assert "LIMIT 50" in guarded.upper()


def test_sql_enforce_max_limit_cap():
    query = "SELECT * FROM orders LIMIT 50000"
    guarded = enforce_query_limits(query, default_limit=50, max_limit=100)
    assert "LIMIT 100" in guarded.upper()


# =====================================================================
# 3. Schema & PII Guardrail Tests
# =====================================================================

def test_pii_sanitization():
    raw_text = (
        "Customer John Doe, SSN: 123-45-6789, CC: 4111 2222 3333 4444, "
        "Email: john.doe@novatech.com requested assistance."
    )
    sanitized = sanitize_pii(raw_text)
    assert "123-45-6789" not in sanitized
    assert "[REDACTED_SSN]" in sanitized
    assert "4111 2222 3333 4444" not in sanitized
    assert "[REDACTED_CC]" in sanitized
    assert "john.doe@novatech.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized


def test_validate_response_schema_valid():
    payload = {
        "status": "APPROVED",
        "refund_amount": 250.00,
        "message": "Refund processed.",
    }
    audit = validate_response_schema(
        payload=payload,
        required_fields=["status", "refund_amount"],
        field_types={"status": str, "refund_amount": float},
    )
    assert audit["is_valid"] is True
    assert len(audit["errors"]) == 0


def test_validate_response_schema_catches_negative_financials():
    invalid_payload = {
        "status": "APPROVED",
        "refund_amount": -50.00,  # Financial invariant violation
        "message": "Refund error.",
    }
    audit = validate_response_schema(
        payload=invalid_payload,
        required_fields=["status", "refund_amount"],
    )
    assert audit["is_valid"] is False
    assert any("negative" in err.lower() for err in audit["errors"])


# =====================================================================
# 4. End-to-End Guarded Pipeline Integration
# =====================================================================

def test_pipeline_safe_turn_execution():
    res = run_memory_guardrails_pipeline(
        session_id="PYTEST-SESSION-01",
        user_message="Show my recent tech orders. Email: alice@example.com",
        sql_query="SELECT order_id, product_name FROM orders WHERE category = 'Laptops'",
    )
    assert res["session_id"] == "PYTEST-SESSION-01"
    assert "[REDACTED_EMAIL]" in res["user_message"]
    assert res["sql_audit"]["status"] == "APPROVED"
    assert "LIMIT 50" in res["sql_audit"]["guarded_sql"].upper()
    assert res["latency_ms"] > 0


def test_pipeline_blocks_sql_injection():
    res = run_memory_guardrails_pipeline(
        session_id="PYTEST-SESSION-02",
        user_message="Reset system table",
        sql_query="DROP TABLE store_faq;",
    )
    assert res["sql_audit"]["status"] == "BLOCKED"
    assert "Destructive operation" in res["sql_audit"]["reason"]
    assert "SECURITY ALERT" in res["final_response"]
