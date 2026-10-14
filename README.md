# 🤖 Agentic AI Playbook: Production Multi-Agent Systems & Advanced RAG

[![CI](https://github.com/chanakyachandu/agentic-ai-playbook/actions/workflows/ci.yml/badge.svg)](https://github.com/chanakyachandu/agentic-ai-playbook/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-18%20Passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![AI-Core](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![Schema](https://img.shields.io/badge/Architecture-100%25%20Function--Based-green.svg)](01_advanced_rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-ready blueprint and reference hub for building, evaluating, and deploying **Agentic AI Systems**, **Autonomous ReAct Agents**, and **Production-Grade Hybrid RAG Pipelines** using Google Gemini, Pydantic, and DuckDB.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Client["🌐 Client & User Interface"]
        REQ["User Request / Goal Prompt"]
        UI["Interactive Walkthrough Notebooks"]
    end

    subgraph Agents["🤖 Autonomous Agent Patterns"]
        REACT["ReAct Loop: Thought ➔ Action ➔ Observation ➔ Answer"]
        PLAN["Plan-and-Solve: Step-by-Step Task Decomposition"]
        REFLECT["Reflexion: Self-Reflection & Policy Correction"]
    end

    subgraph Tools["🛠️ Real-World Tool Execution"]
        ORDER["Order Database Lookup (orders_database.json)"]
        POLICY["Category Policy Validation (Laptops, Drones, Apparel)"]
        CALC["Refund Math & Restocking Fee Deductions"]
    end

    subgraph RAG["🔍 Advanced Hybrid RAG Engine"]
        CHUNK["Text Cleaning & Section-Aware Chunking"]
        HYBRID["Hybrid Search: BM25 Lexical + Dense Vectors (RRF)"]
        RERANK["Re-Ranking & Verifiable Citation Extraction"]
    end

    subgraph Safety["🛡️ Enterprise Financial Safety Gate"]
        HITL["Human-in-the-Loop (HITL): Auto-approves <= $500, Pauses > $500"]
    end

    REQ --> REACT
    REQ --> PLAN
    REACT --> Tools
    PLAN --> Tools
    Tools --> HITL
    REQ --> RAG
    RAG --> UI
    HITL --> UI
```

---

## 📁 Repository Structure

```
agentic-ai-playbook/
├── 01_advanced_rag/                  # 🔍 RAG Foundations & Retrieval Engines
│   ├── chunking_strategies.py        # Text cleaning (removes ===/---) & section chunking
│   ├── hybrid_retriever.py           # BM25 + Dense Vectors with Reciprocal Rank Fusion (RRF)
│   ├── reranker.py                   # Proximity re-ranking & verifiable citation extraction
│   └── rag_pipeline.py               # End-to-end grounded query execution
│
├── 02_agentic_patterns/              # 🤖 Core Agent Reasoning Loops & Safety
│   ├── tools.py                      # Order lookup, policy validation, refund calculation & HITL gate
│   ├── react_agent.py                # ReAct (Thought ➔ Action ➔ Observation) with loop-breaker (max_steps=5)
│   └── plan_and_solve.py             # Plan-and-Execute decomposition & Reflexion self-critique
│
├── 03_multiagent_systems/            # 🐝 Multi-Agent Swarms & Deep Research
├── 04_memory_and_guardrails/         # 🛡️ Production Safety & State
├── 05_serving_and_evals/             # ⚡ Production Serving & LLM CI/CD
│
├── data/ecommerce/                   # 🛒 Real-world policies & customer orders database
│   ├── 2026_current_policy.txt       # Store return, warranty, and VIP policies
│   ├── orders_database.json          # Realistic multi-category customer orders
│   └── store_faq_kb.txt              # Customer service FAQ runbook
│
├── notebooks/                        # 📓 Interactive Walkthroughs
│   ├── 01_advanced_rag_walkthrough.ipynb
│   └── 02_agentic_patterns_walkthrough.ipynb
│
├── tests/                            # 🧪 Pytest verification suite (18 Passing)
│   ├── test_rag.py                   # 10 tests verifying chunking, BM25, dense, RRF, citations
│   ├── test_agents.py                # 8 tests verifying tools, ReAct loop, HITL gate, plan-and-solve
│   └── test_setup.py                 # Smoke tests for package discovery
│
├── .github/workflows/ci.yml          # ⚙️ GitHub Actions automated testing (Py 3.10 - 3.12)
├── requirements.txt                  # Standard dependencies
└── .env.example                      # Configuration template
```

---

## ⚡ Quick Start

### 1. Clone & Setup Environment
```bash
git clone https://github.com/chanakyachandu/agentic-ai-playbook.git
cd agentic-ai-playbook

# Virtual environment setup
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys
Copy the environment template and insert your Gemini API key:
```bash
cp .env.example .env
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📋 Resume Highlights

> - **Production Multi-Agent Swarms**: Architected autonomous multi-agent pipelines with hierarchical Supervisor coordination, dynamic tool execution, and consensus-driven fact-checking.
> - **Enterprise RAG & Hybrid Retrieval**: Implemented hybrid search combining BM25 keyword matching and dense embeddings with Reciprocal Rank Fusion (RRF) and citation grounding.
> - **Defensive Guardrails & Memory**: Engineered AST-based SQL guardrails to block destructive queries and stateful sliding-window session memory for multi-turn conversations.
> - **High-Performance Serving & CI/CD**: Built streaming FastAPI endpoints with Server-Sent Events (SSE) and automated GitHub Actions CI testing across Python 3.10–3.12.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
