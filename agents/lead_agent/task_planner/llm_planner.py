"""LLM-guided investigation planner in strict mode."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from agents.lead_agent.task_planner.types import InvestigationPlan, SubTask
from app.investigation_errors import PlannerLLMError

_MODEL = os.environ.get("AGENTIC_PLANNER_MODEL", "llama-3.1-8b-instant")

_DEFAULT_TOOLS_BY_AGENT: Dict[str, List[str]] = {
    "corporate_agent": ["sec_edgar"],
    "legal_agent": ["ofac", "courtlistener"],
    "social_graph_agent": ["gdelt"],
}


def _available_tools_prompt(available_tools_by_agent: Dict[str, Iterable[str]]) -> str:
    lines = []
    for agent_name, tool_names in available_tools_by_agent.items():
        lines.append(f"- {agent_name}: {', '.join(tool_names)}")
    return "\n".join(lines)


def _build_prompt(
    query: str,
    entity: Optional[Any],
    available_tools_by_agent: Dict[str, Iterable[str]],
) -> str:
    entity_name = getattr(entity, "name", None) or "Unknown entity"
    identifiers = getattr(entity, "identifiers", {}) or {}
    entity_type = getattr(entity, "entity_type", "unknown")
    return (
        "You are the Lead Investigation Planner for a financial-risk OSINT system.\n"
        "Your role is to produce a complete and executable plan from a user query.\n\n"
        "You must:\n"
        "- Build testable hypotheses grounded in available public data sources.\n"
        "- Cover corporate, legal, and network/media evidence lanes unless clearly irrelevant.\n"
        "- Keep execution bounded (no open-ended research loops).\n"
        "- Use only the provided agents and tools.\n\n"
        "Output contract:\n"
        "- Return VALID JSON ONLY.\n"
        "- No markdown, no narrative wrapper, no comments.\n"
        "- Never return an empty tasks list.\n"
        "- Every task must include task_type, target_agent, candidate_tools, priority, rationale.\n\n"
        "Produce JSON with this schema:\n"
        "{\n"
        '  "investigation_goal": string,\n'
        '  "hypotheses": [string],\n'
        '  "tasks": [\n'
        "    {\n"
        '      "task_type": string,\n'
        '      "target_agent": "corporate_agent" | "legal_agent" | "social_graph_agent",\n'
        '      "description": string,\n'
        '      "candidate_tools": [string],\n'
        '      "priority": "high" | "medium" | "low",\n'
        '      "rationale": string\n'
        "    }\n"
        "  ],\n"
        '  "success_criteria": [string],\n'
        '  "max_rounds": integer\n'
        "}\n\n"
        "Audit posture: evidence-first, cautious, traceable, and non-legal.\n"
        "Keep the plan bounded and tool-aware. Prefer 3-6 tasks and at most 2 rounds.\n"
        "Only use the following tools per agent:\n"
        f"{_available_tools_prompt(available_tools_by_agent)}\n\n"
        f"Query: {query}\n"
        f"Entity name: {entity_name}\n"
        f"Entity type: {entity_type}\n"
        f"Identifiers: {json.dumps(identifiers, sort_keys=True)}\n"
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty planner output")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("planner output did not contain JSON")
    return json.loads(text[start : end + 1])


def _normalize_target_agent(
    raw_target_agent: str,
    *,
    task_type: str,
    description: str,
    candidate_tools: Any,
    available_tools_by_agent: Dict[str, Iterable[str]],
) -> str:
    normalized = (raw_target_agent or "").strip().lower().replace("-", "_").replace(" ", "_")
    valid_agents = {"corporate_agent", "legal_agent", "social_graph_agent"}
    if normalized in valid_agents:
        return normalized

    alias_map = {
        "corporate": "corporate_agent",
        "corp": "corporate_agent",
        "company": "corporate_agent",
        "corporate_lane": "corporate_agent",
        "legal": "legal_agent",
        "compliance": "legal_agent",
        "regulatory": "legal_agent",
        "legal_lane": "legal_agent",
        "social": "social_graph_agent",
        "social_graph": "social_graph_agent",
        "media": "social_graph_agent",
        "network": "social_graph_agent",
        "social_graph_lane": "social_graph_agent",
    }
    if normalized in alias_map:
        return alias_map[normalized]

    tool_to_agent: Dict[str, str] = {}
    for agent_name, tools in available_tools_by_agent.items():
        for tool in tools:
            tool_to_agent[str(tool).strip().lower()] = agent_name
    if isinstance(candidate_tools, list):
        for tool in candidate_tools:
            inferred = tool_to_agent.get(str(tool).strip().lower())
            if inferred:
                return inferred

    task_context = f"{task_type} {description}".lower()
    if any(word in task_context for word in ("sec", "filing", "ownership", "governance", "board")):
        return "corporate_agent"
    if any(word in task_context for word in ("sanction", "legal", "court", "docket", "compliance", "enforcement")):
        return "legal_agent"
    if any(word in task_context for word in ("news", "media", "network", "graph", "reputation", "adverse")):
        return "social_graph_agent"

    raise PlannerLLMError(f"planner returned unsupported target_agent: {raw_target_agent}")


def _coerce_task(
    raw: Dict[str, Any],
    *,
    available_tools_by_agent: Dict[str, Iterable[str]],
) -> SubTask:
    task_type = str(raw.get("task_type") or "").strip()
    target_agent_raw = str(raw.get("target_agent") or "").strip()
    description = str(raw.get("description") or "").strip()
    priority = str(raw.get("priority") or "").strip().lower()
    rationale = str(raw.get("rationale") or "").strip()
    candidate_tools = raw.get("candidate_tools")
    if not task_type or not target_agent_raw or not description:
        raise PlannerLLMError("planner output is missing required task fields")
    target_agent = _normalize_target_agent(
        target_agent_raw,
        task_type=task_type,
        description=description,
        candidate_tools=candidate_tools,
        available_tools_by_agent=available_tools_by_agent,
    )
    if priority not in {"high", "medium", "low"}:
        raise PlannerLLMError(f"planner returned invalid task priority: {priority}")
    if not isinstance(candidate_tools, list) or not candidate_tools:
        raise PlannerLLMError("planner task must include a non-empty candidate_tools array")
    allowed = set(str(name) for name in available_tools_by_agent.get(target_agent, ()))
    normalized_tools = tuple(str(t).strip() for t in candidate_tools if str(t).strip())
    normalized_tools = tuple(name for name in normalized_tools if name in allowed)
    if not normalized_tools:
        raise PlannerLLMError(
            f"planner task '{task_type}' for '{target_agent}' has no candidate tools available in current runtime"
        )
    return SubTask(
        task_type=task_type,
        target_agent=target_agent,
        description=description,
        candidate_tools=normalized_tools,
        priority=priority,
        rationale=rationale,
        origin="llm_planner",
    )


def _validate_plan(
    raw_plan: Dict[str, Any],
    *,
    query: str,
    available_tools_by_agent: Dict[str, Iterable[str]],
) -> InvestigationPlan:
    tasks = []
    for item in (raw_plan.get("tasks") or []):
        if not isinstance(item, dict):
            continue
        tasks.append(
            _coerce_task(
                item,
                available_tools_by_agent=available_tools_by_agent,
            )
        )
    if not tasks:
        raise PlannerLLMError("planner returned no tasks")

    max_rounds_raw = raw_plan.get("max_rounds")
    if not isinstance(max_rounds_raw, int):
        raise PlannerLLMError("planner must return integer max_rounds")
    max_rounds = max(1, min(max_rounds_raw, 2))
    investigation_goal = str(raw_plan.get("investigation_goal") or "").strip()
    if not investigation_goal:
        investigation_goal = query.strip()
    if not investigation_goal:
        raise PlannerLLMError("planner returned empty investigation_goal")
    return InvestigationPlan(
        investigation_goal=investigation_goal,
        hypotheses=[str(item).strip() for item in (raw_plan.get("hypotheses") or []) if str(item).strip()],
        tasks=tasks,
        success_criteria=[str(item).strip() for item in (raw_plan.get("success_criteria") or []) if str(item).strip()],
        max_rounds=max_rounds,
        planner="llm",
        planner_notes="LLM-generated structured plan.",
    )


def _call_llm(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise PlannerLLMError("planner failed: GROQ_API_KEY is not set")
    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=30.0)
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise PlannerLLMError("planner failed: empty LLM response")
        return text.strip()
    except PlannerLLMError:
        raise
    except Exception as exc:
        raise PlannerLLMError(f"planner LLM request failed: {exc}") from exc


def plan_investigation(
    query: str,
    *,
    entity: Optional[Any] = None,
    available_tools_by_agent: Optional[Dict[str, Iterable[str]]] = None,
    llm_client: Optional[Any] = None,
) -> InvestigationPlan:
    """Return an LLM-guided plan with strict validation."""
    available_tools_by_agent = available_tools_by_agent or _DEFAULT_TOOLS_BY_AGENT
    prompt = _build_prompt(query, entity, available_tools_by_agent)

    raw_text: Optional[str]
    if llm_client is not None:
        raw_text = llm_client(prompt)
        if not raw_text or not str(raw_text).strip():
            raise PlannerLLMError("planner failed: test/client returned empty response")
    else:
        raw_text = _call_llm(prompt)

    try:
        raw_plan = _extract_json(raw_text)
    except Exception as first_exc:
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous output was invalid JSON. "
            "Rewrite and return ONLY valid JSON matching the schema exactly.\n"
            f"Invalid output was:\n{raw_text}"
        )
        retry_text = llm_client(retry_prompt) if llm_client is not None else _call_llm(retry_prompt)
        if not retry_text or not str(retry_text).strip():
            raise PlannerLLMError(f"planner returned invalid JSON: {first_exc}") from first_exc
        try:
            raw_plan = _extract_json(str(retry_text))
        except Exception as retry_exc:
            raise PlannerLLMError(f"planner returned invalid JSON: {retry_exc}") from retry_exc

    try:
        return _validate_plan(
            raw_plan,
            query=query,
            available_tools_by_agent=available_tools_by_agent,
        )
    except PlannerLLMError as first_validation_exc:
        repair_prompt = (
            f"{prompt}\n\n"
            "Your previous JSON failed schema validation in runtime.\n"
            f"Validation error: {first_validation_exc}\n"
            "Rewrite and return ONLY corrected JSON using valid target_agent values and available tools."
        )
        repair_text = llm_client(repair_prompt) if llm_client is not None else _call_llm(repair_prompt)
        if not repair_text or not str(repair_text).strip():
            raise
        try:
            repaired = _extract_json(str(repair_text))
            return _validate_plan(
                repaired,
                query=query,
                available_tools_by_agent=available_tools_by_agent,
            )
        except Exception:
            raise first_validation_exc
