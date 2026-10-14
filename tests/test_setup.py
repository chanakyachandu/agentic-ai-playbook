"""
Smoke test verifying repository package discovery and environment structure.
"""
import os
import importlib

def test_package_structure():
    """Assert all core modular packages are importable."""
    packages = [
        "01_advanced_rag",
        "02_agentic_patterns",
        "03_multiagent_systems",
        "04_memory_and_guardrails",
        "05_serving_and_evals",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None

def test_env_template_exists():
    """Assert .env.example exists."""
    assert os.path.exists(".env.example")
