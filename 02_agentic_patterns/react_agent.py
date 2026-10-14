"""
Autonomous ReAct (Reason + Act) Agent.
100% Function-Based: No classes, pure Python functions with loop-breaker protection.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

try:
    from .tools import get_tools_registry, execute_tool
except ImportError:
    from tools import get_tools_registry, execute_tool

load_dotenv()


def format_tools_prompt(registry: Dict[str, Any]) -> str:
    """Formats available tools descriptions for the agent prompt."""
    lines = []
    for name, item in registry.items():
        lines.append(f"- {name}: {item['description']}")
    return "\n".join(lines)


def parse_action_call(action_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses tool action syntax like:
    'lookup_order(order_id="ORD-2026-1001")'
    or JSON action format.
    """
    clean = action_text.strip()
    match = re.match(r"^(\w+)\((.*)\)$", clean, re.DOTALL)
    if not match:
        return None

    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()

    # Parse key=value arguments
    kwargs = {}
    if args_str:
        # Match arg_name="string" or arg_name=123.45 or arg_name=True
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s,]+))'
        for k, s1, s2, raw in re.findall(pattern, args_str):
            val = s1 or s2
            if val is None or val == "":
                if raw.lower() in ("true", "false"):
                    val = raw.lower() == "true"
                else:
                    try:
                        val = float(raw) if "." in raw else int(raw)
                    except ValueError:
                        val = raw
            kwargs[k] = val

    return {"tool_name": tool_name, "kwargs": kwargs}


def run_react_agent(
    user_query: str,
    max_steps: int = 5,
    api_key: Optional[str] = None,
    tools_registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes the ReAct reasoning loop (Thought -> Action -> Observation -> Final Answer).
    Pure function with max_steps safety gate to prevent infinite loops.
    """
    if tools_registry is None:
        tools_registry = get_tools_registry()

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    steps_trace: List[Dict[str, Any]] = []
    hitl_flag = False

    # Check for live Gemini API
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            system_instruction = (
                "You are an enterprise customer service AI agent solving user inquiries using the ReAct loop.\n"
                "You have access to the following tools:\n"
                f"{format_tools_prompt(tools_registry)}\n\n"
                "Use the exact format:\n"
                "Thought: [Explain your reasoning on what to do next]\n"
                "Action: [tool_name(param1=\"value\", ...)]\n"
                "Observation: [Tool result will be inserted here]\n"
                "... (Repeat Thought/Action/Observation as needed)\n"
                "Thought: [Explain final conclusion]\n"
                "Final Answer: [Your complete response to the user]\n"
            )

            conversation_history = f"User Inquiry: {user_query}\n"

            for step_num in range(1, max_steps + 1):
                # Prompt model for next thought/action
                prompt = f"{system_instruction}\n\n{conversation_history}"
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                output_text = response.text.strip() if response and response.text else ""

                # Check if Final Answer is reached
                if "Final Answer:" in output_text:
                    final_part = output_text.split("Final Answer:", 1)[1].strip()
                    thought_part = output_text.split("Final Answer:", 1)[0].replace("Thought:", "").strip()
                    steps_trace.append({
                        "step": step_num,
                        "thought": thought_part or "Formulating final conclusion.",
                        "action": "COMPLETED",
                        "observation": "N/A",
                    })
                    return {
                        "query": user_query,
                        "steps": steps_trace,
                        "final_answer": final_part,
                        "hitl_triggered": hitl_flag,
                        "total_steps": step_num,
                    }

                # Parse Thought and Action
                thought = "Reasoning on current state."
                action_str = ""
                if "Thought:" in output_text:
                    thought = output_text.split("Action:")[0].replace("Thought:", "").strip()
                if "Action:" in output_text:
                    action_str = output_text.split("Action:")[1].split("\n")[0].strip()

                parsed = parse_action_call(action_str)
                if parsed:
                    obs = execute_tool(parsed["tool_name"], parsed["kwargs"], registry=tools_registry)
                    obs_str = json.dumps(obs)
                    if obs.get("status") == "REQUIRES_HUMAN_APPROVAL":
                        hitl_flag = True

                    steps_trace.append({
                        "step": step_num,
                        "thought": thought,
                        "action": action_str,
                        "observation": obs_str,
                    })
                    conversation_history += f"Thought: {thought}\nAction: {action_str}\nObservation: {obs_str}\n"
                else:
                    break
        except Exception:
            # Fall through to deterministic offline execution
            pass

    # ==============================================================================
    # Deterministic Offline Execution Engine (Safe, Zero-Network-Dependency)
    # ==============================================================================
    # Detect Order ID in query (e.g. ORD-2026-1001)
    order_match = re.search(r"ORD-\d{4}-\d{4}", user_query.upper())
    target_order_id = order_match.group(0) if order_match else "ORD-2026-1001"

    # Step 1: Look up Order
    t1 = f"I need to check details and purchase date for order {target_order_id}."
    a1 = f'lookup_order(order_id="{target_order_id}")'
    o1 = execute_tool("lookup_order", {"order_id": target_order_id}, registry=tools_registry)
    steps_trace.append({"step": 1, "thought": t1, "action": a1, "observation": json.dumps(o1)})

    if o1.get("status") != "FOUND":
        return {
            "query": user_query,
            "steps": steps_trace,
            "final_answer": f"Unable to locate order {target_order_id}. Please verify the order number.",
            "hitl_triggered": False,
            "total_steps": 1,
        }

    category = o1.get("category", "General")
    unit_price = o1.get("unit_price_usd", 100.0)

    # Step 2: Check Policy
    t2 = f"Now checking return policy rules for category '{category}'."
    a2 = f'check_return_policy(category="{category}")'
    o2 = execute_tool("check_return_policy", {"category": category}, registry=tools_registry)
    steps_trace.append({"step": 2, "thought": t2, "action": a2, "observation": json.dumps(o2)})

    # Step 3: Calculate Refund
    fee_pct = o2.get("restocking_fee_pct", 0.15)
    t3 = f"Calculating refund amount for ${unit_price} with {fee_pct*100:.0f}% restocking fee."
    a3 = f'calculate_refund(unit_price={unit_price}, restocking_fee_pct={fee_pct})'
    o3 = execute_tool("calculate_refund", {"unit_price": unit_price, "restocking_fee_pct": fee_pct}, registry=tools_registry)
    steps_trace.append({"step": 3, "thought": t3, "action": a3, "observation": json.dumps(o3)})

    net_refund = o3.get("net_refund_usd", unit_price)

    # Step 4: Process Refund with HITL constraint
    t4 = f"Submitting refund request of ${net_refund:.2f}."
    a4 = f'process_refund(order_id="{target_order_id}", refund_amount={net_refund}, reason="Customer Return Request")'
    o4 = execute_tool("process_refund", {"order_id": target_order_id, "refund_amount": net_refund, "reason": "Customer Return Request"}, registry=tools_registry)
    if o4.get("status") == "REQUIRES_HUMAN_APPROVAL":
        hitl_flag = True

    steps_trace.append({"step": 4, "thought": t4, "action": a4, "observation": json.dumps(o4)})

    # Formulate Final Answer
    if hitl_flag:
        final_answer = (
            f"Order {target_order_id} ({o1['item_name']}) is eligible for return under {o2['policy_clause']}. "
            f"Calculated net refund is ${net_refund:.2f} after a ${o3['restocking_fee_usd']:.2f} restocking fee. "
            f"⚠️ HUMAN APPROVAL REQUIRED: Because ${net_refund:.2f} exceeds the $500 auto-approval limit, "
            f"the request has been routed to customer operations manager for final authorization."
        )
    else:
        final_answer = (
            f"Order {target_order_id} has been processed for refund. "
            f"Net amount of ${net_refund:.2f} has been approved back to your original payment method."
        )

    return {
        "query": user_query,
        "steps": steps_trace,
        "final_answer": final_answer,
        "hitl_triggered": hitl_flag,
        "total_steps": len(steps_trace),
    }
