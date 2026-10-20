"""
Schema & PII Guardrails for Enterprise Agent Systems.
100% Function-Based: Zero classes, pure composable Python functions.

Capabilities:
- PII Sanitization: Redacts Credit Card numbers, SSNs, and email addresses.
- Schema Invariant Verification: Checks required keys, data types, and non-negative constraints.
- Output Safety Audit: Prevents leaking sensitive data or returning malformed LLM outputs to users.
"""

import re
from typing import Dict, Any, List, Optional, Type


# Regex patterns for sensitive PII
CC_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def sanitize_pii(text: str) -> str:
    """
    Redacts sensitive personal identifiable information (PII) from strings:
    - Credit Card numbers -> [REDACTED_CC]
    - Social Security numbers -> [REDACTED_SSN]
    - Email addresses -> [REDACTED_EMAIL]
    """
    if not isinstance(text, str) or not text:
        return text

    sanitized = CC_REGEX.sub("[REDACTED_CC]", text)
    sanitized = SSN_REGEX.sub("[REDACTED_SSN]", sanitized)
    sanitized = EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized)
    return sanitized


def sanitize_payload_pii(data: Any) -> Any:
    """
    Recursively scans and sanitizes PII across dictionaries, lists, and strings.
    """
    if isinstance(data, str):
        return sanitize_pii(data)
    elif isinstance(data, dict):
        return {k: sanitize_payload_pii(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_payload_pii(item) for item in data]
    return data


def validate_response_schema(
    payload: Dict[str, Any],
    required_fields: List[str],
    field_types: Optional[Dict[str, Type]] = None
) -> Dict[str, Any]:
    """
    Audits structured LLM dictionary output against required keys, types, and values.
    
    Returns:
        Dict containing:
        - is_valid: bool
        - errors: List[str]
        - sanitized_payload: Dict[str, Any]
    """
    if not isinstance(payload, dict):
        return {
            "is_valid": False,
            "errors": [f"Expected dictionary payload, received {type(payload).__name__}."],
            "sanitized_payload": {},
        }

    errors = []

    # 1. Required fields check
    for field in required_fields:
        if field not in payload or payload[field] is None:
            errors.append(f"Missing mandatory field: '{field}'")

    # 2. Field types validation
    if field_types:
        for field, expected_type in field_types.items():
            if field in payload and payload[field] is not None:
                if not isinstance(payload[field], expected_type):
                    errors.append(
                        f"Field '{field}' expected type '{expected_type.__name__}', "
                        f"got '{type(payload[field]).__name__}'."
                    )

    # 3. Numeric bounds invariant check (amounts cannot be negative)
    for k, v in payload.items():
        if isinstance(v, (int, float)) and ("amount" in k.lower() or "fee" in k.lower() or "refund" in k.lower()):
            if v < 0:
                errors.append(f"Financial invariant violated: '{k}' cannot be negative (value: {v}).")

    # 4. Sanitize all string fields for PII
    sanitized = sanitize_payload_pii(payload)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "sanitized_payload": sanitized,
    }
