"""
Memory Manager for Multi-Turn Agent Conversations.
100% Function-Based: Pure, composable Python functions with zero class boilerplate.

Features:
- Sliding-Window History: Retains the latest N message turns to prevent prompt bloat.
- Rolling Summary Memory: Compresses older conversation turns to preserve context within a small token budget.
- Combined Memory Context: Builds formatted prompt injection blocks for LLM reasoning.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    for cand in [
        os.path.join("..", ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"),
    ]:
        if os.path.exists(cand):
            load_dotenv(cand)
            break


def create_memory_session(session_id: str, max_window: int = 4) -> Dict[str, Any]:
    """
    Initializes a structured session memory store.
    """
    return {
        "session_id": session_id,
        "max_window": max_window,
        "messages": [],
        "rolling_summary": "",
        "turn_count": 0,
    }


def add_message(session: Dict[str, Any], role: str, content: str) -> Dict[str, Any]:
    """
    Appends a new turn (role: 'user', 'assistant', 'system') to the session memory.
    """
    msg = {
        "turn": session["turn_count"] + 1,
        "role": role,
        "content": content.strip(),
    }
    session["messages"].append(msg)
    session["turn_count"] += 1
    return session


def get_sliding_window(session: Dict[str, Any], window_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns only the most recent N messages, discarding older turns from the prompt payload.
    """
    w = window_size if window_size is not None else session.get("max_window", 4)
    if w <= 0:
        return []
    return session["messages"][-w:]


def summarize_history(session: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """
    Summarizes messages that have slid out of the active window.
    Uses Gemini when configured, or provides deterministic extractive compression.
    """
    w = session.get("max_window", 4)
    messages = session.get("messages", [])
    
    if len(messages) <= w:
        return session.get("rolling_summary", "")

    older_messages = messages[:-w]
    if not older_messages:
        return session.get("rolling_summary", "")

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in older_messages
    )

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "You are an expert conversation summarizer for customer service agents.\n"
                f"Existing summary: {session.get('rolling_summary', 'None')}\n\n"
                f"New older messages to incorporate:\n{conversation_text}\n\n"
                "Provide a concise 2-sentence summary capturing customer intent, orders, and key issues."
            )
            resp = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            if resp and resp.text:
                summary = resp.text.strip()
                session["rolling_summary"] = summary
                return summary
        except Exception:
            pass

    # Deterministic extractive summary fallback (reliable, offline)
    key_points = []
    if session.get("rolling_summary"):
        key_points.append(session["rolling_summary"])
    for m in older_messages[-3:]:
        key_points.append(f"{m['role']}: {m['content'][:60]}...")
    summary = " | ".join(key_points)
    session["rolling_summary"] = summary
    return summary


def build_memory_context(session: Dict[str, Any]) -> str:
    """
    Builds the combined prompt injection context containing:
    1. Rolling Summary of past history (if any)
    2. Active Sliding Window turns
    """
    window_msgs = get_sliding_window(session)
    sections = []

    if session.get("rolling_summary"):
        sections.append(f"[PREVIOUS CONTEXT SUMMARY]\n{session['rolling_summary']}")

    if window_msgs:
        turn_lines = [f"{m['role'].upper()}: {m['content']}" for m in window_msgs]
        sections.append("[ACTIVE CONVERSATION TURNS]\n" + "\n".join(turn_lines))

    return "\n\n".join(sections) if sections else "[NO PREVIOUS HISTORY]"


def clear_memory_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resets the session conversation history and summary.
    """
    session["messages"] = []
    session["rolling_summary"] = ""
    session["turn_count"] = 0
    return session
