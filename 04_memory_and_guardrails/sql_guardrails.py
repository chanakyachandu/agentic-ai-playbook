"""
SQL AST Guardrails for Agentic Text-to-SQL Systems.
100% Function-Based: Zero classes, pure composable Python functions.

Capabilities:
- AST Parsing via sqlglot: Deep syntax analysis catching obfuscated injections.
- Destructive Expression Blocking: Forbids DROP, DELETE, TRUNCATE, ALTER, UPDATE, INSERT, CREATE.
- Multi-Statement Rejection: Disallows stacked injection attacks (e.g. 'SELECT 1; DROP TABLE users').
- Row Limit Enforcement: Automatically injects or caps LIMIT clauses to protect database memory.
"""

from typing import Dict, Any, Tuple
import sqlglot
from sqlglot import exp


# Disallowed SQL AST expression classes
FORBIDDEN_AST_TYPES = (
    exp.Drop,
    exp.Delete,
    exp.TruncateTable,
    exp.Alter,
    exp.Update,
    exp.Insert,
    exp.Create,
    exp.Command,
)

FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT", "CREATE", "GRANT", "REVOKE"
]


def validate_sql_query(query: str, dialect: str = "duckdb") -> Dict[str, Any]:
    """
    Parses and audits a SQL query using Abstract Syntax Tree (AST) validation.
    
    Returns:
        Dict containing:
        - is_safe: bool
        - ast_type: str
        - reason: str or None
        - sanitized_sql: str
    """
    clean_query = query.strip().rstrip(";")
    if not clean_query:
        return {
            "is_safe": False,
            "ast_type": "EMPTY",
            "reason": "Query string is empty.",
            "sanitized_sql": "",
        }

    # 1. Check for multiple stacked statements (common SQL injection pattern)
    try:
        parsed_statements = sqlglot.parse(clean_query, read=dialect)
    except Exception as e:
        return {
            "is_safe": False,
            "ast_type": "SYNTAX_ERROR",
            "reason": f"SQL syntax parsing error: {str(e)}",
            "sanitized_sql": clean_query,
        }

    if len(parsed_statements) > 1:
        return {
            "is_safe": False,
            "ast_type": "MULTI_STATEMENT",
            "reason": "Multiple stacked SQL statements detected. Only single read-only queries are permitted.",
            "sanitized_sql": clean_query,
        }

    expression = parsed_statements[0]
    if expression is None:
        return {
            "is_safe": False,
            "ast_type": "EMPTY",
            "reason": "Parsed expression is null.",
            "sanitized_sql": clean_query,
        }

    # 2. Check for forbidden AST nodes anywhere in the syntax tree
    for node in expression.walk():
        if isinstance(node, FORBIDDEN_AST_TYPES):
            node_name = node.__class__.__name__.upper()
            return {
                "is_safe": False,
                "ast_type": node_name,
                "reason": f"Destructive operation detected: '{node_name}' is strictly forbidden in read-only mode.",
                "sanitized_sql": clean_query,
            }

    # 3. Ensure root expression is a SELECT or UNION of SELECTs
    if not isinstance(expression, (exp.Select, exp.Union)):
        return {
            "is_safe": False,
            "ast_type": expression.__class__.__name__.upper(),
            "reason": f"Only SELECT queries are permitted. Found root node '{expression.__class__.__name__}'.",
            "sanitized_sql": clean_query,
        }

    return {
        "is_safe": True,
        "ast_type": "SELECT",
        "reason": None,
        "sanitized_sql": expression.sql(dialect=dialect),
    }


def enforce_query_limits(
    query: str,
    default_limit: int = 50,
    max_limit: int = 100,
    dialect: str = "duckdb"
) -> str:
    """
    Audits and enforces a safe LIMIT clause on SELECT queries.
    - If no LIMIT exists: Appends LIMIT default_limit.
    - If LIMIT > max_limit: Overwrites with max_limit to prevent memory exhaustion.
    """
    try:
        parsed = sqlglot.parse_one(query.strip().rstrip(";"), read=dialect)
    except Exception:
        # If unparseable, return original query safely
        return query

    if not isinstance(parsed, (exp.Select, exp.Union)):
        return query

    limit_node = parsed.args.get("limit")

    if limit_node is None:
        # Automatically attach default limit
        parsed = parsed.limit(default_limit)
    else:
        # Check current limit value
        try:
            curr_val = int(limit_node.expression.this)
            if curr_val > max_limit:
                parsed = parsed.limit(max_limit)
        except Exception:
            parsed = parsed.limit(default_limit)

    return parsed.sql(dialect=dialect)
