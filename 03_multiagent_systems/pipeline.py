"""
Main Multi-Agent Systems Integration Pipeline.
100% Function-Based Architecture: Zero classes, pure composable Python functions.

Integrates all specialist workers orchestrated by the Supervisor:
  - Worker 1: Retrieval Worker (Researcher) -> retrieval_worker.py
  - Worker 2: Tool Worker (Accountant) -> tool_worker.py
  - Worker 3: Compliance Worker (Auditor) -> compliance_worker.py
  - Coordinator: Supervisor Agent (Boss) -> supervisor_agent.py
"""

import sys
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    from .supervisor_agent import run_supervisor
    from .retrieval_worker import run_retrieval_worker
    from .tool_worker import run_tool_worker
    from .compliance_worker import run_compliance_worker
except ImportError:
    from supervisor_agent import run_supervisor
    from retrieval_worker import run_retrieval_worker
    from tool_worker import run_tool_worker
    from compliance_worker import run_compliance_worker

load_dotenv()


def run_multiagent_pipeline(
    query: str,
    auto_approve_limit: float = 500.0,
    api_key: Optional[str] = None,
    data_dir: str = "data/ecommerce",
) -> Dict[str, Any]:
    """
    Main Integrated Multi-Agent Pipeline.
    Takes a raw customer inquiry, orchestrates the Supervisor Swarm,
    enforces invariant audits & financial safety gates, and produces an auditable trace.
    """
    start_time = time.time()

    # Execute supervisor swarm
    swarm_result = run_supervisor(
        query=query,
        api_key=api_key,
        data_dir=data_dir,
        auto_approve_limit=auto_approve_limit,
    )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query": query,
        "decision": swarm_result["decision"],
        "hitl_triggered": swarm_result["hitl_triggered"],
        "resolution": swarm_result["final_resolution"],
        "worker_outputs": swarm_result["worker_outputs"],
        "decomposition": swarm_result["decomposition"],
        "latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    sample_query = (
        "Hi, I bought a MacBook Pro (order ORD-2026-1001). I used it for about 6 hours. "
        "I want to return it for a refund. Am I eligible and what is the restocking fee?"
    )
    if len(sys.argv) > 1:
        sample_query = " ".join(sys.argv[1:])

    print("Running Master Multi-Agent Pipeline...")
    res = run_multiagent_pipeline(sample_query)
    print(f"\nQuery: {res['query']}")
    print(f"\nDecision: {res['decision']}")
    print(f"HITL Triggered: {res['hitl_triggered']}")
    print(f"\nResolution:\n{res['resolution']}")
    print(f"\nLatency: {res['latency_ms']} ms")
