"""Tests for strict LLM-only task planner."""

import pytest

from agents.lead_agent.task_planner import InvestigationPlan, build_plan, plan_investigation
from app.investigation_errors import PlannerLLMError
from osint_swarm.entities import Entity


def test_plan_investigation_accepts_mock_llm_json():
    def fake_llm(_prompt: str) -> str:
        return """
        {
          "investigation_goal": "Investigate Acme",
          "hypotheses": ["Acme has legal exposure"],
          "tasks": [
            {
              "task_type": "litigation",
              "target_agent": "legal_agent",
              "description": "Review court records",
              "candidate_tools": ["courtlistener"],
              "priority": "high",
              "rationale": "Court records may reveal enforcement activity"
            }
          ],
          "success_criteria": ["Find legal evidence or document absence"],
          "max_rounds": 2
        }
        """

    plan = plan_investigation("Investigate Acme", llm_client=fake_llm)
    assert plan.planner == "llm"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].candidate_tools == ("courtlistener",)
    assert plan.tasks[0].origin == "llm_planner"


def test_plan_investigation_invalid_llm_output_raises():
    with pytest.raises(PlannerLLMError):
        plan_investigation("Investigate Acme", llm_client=lambda _prompt: "not json")


def test_plan_investigation_missing_tasks_raises():
    with pytest.raises(PlannerLLMError):
        plan_investigation(
            "Investigate Acme",
            llm_client=lambda _prompt: '{"investigation_goal":"x","tasks":[],"max_rounds":1}',
        )


def test_build_plan_uses_strict_llm_client():
    entity = Entity(entity_id="e1", name="Acme Corp", identifiers={"cik": "0000000001"})
    plan = build_plan(
        "Investigate Acme Corp",
        entity=entity,
        llm_client=lambda _prompt: """
        {
          "investigation_goal": "Investigate Acme Corp",
          "hypotheses": ["Acme may have legal exposure"],
          "tasks": [
            {
              "task_type": "litigation",
              "target_agent": "legal_agent",
              "description": "Review court records",
              "candidate_tools": ["courtlistener"],
              "priority": "high",
              "rationale": "Court records may reveal enforcement activity"
            }
          ],
          "success_criteria": ["Review legal lane coverage"],
          "max_rounds": 1
        }
        """,
    )
    assert isinstance(plan, InvestigationPlan)
    assert plan.planner == "llm"


def test_plan_investigation_filters_candidate_tools_to_available_set():
    plan = plan_investigation(
        "Investigate Acme Corp",
        available_tools_by_agent={
            "corporate_agent": ["sec_edgar"],
            "legal_agent": ["ofac", "courtlistener"],
            "social_graph_agent": ["gdelt"],
        },
        llm_client=lambda _prompt: """
        {
          "investigation_goal": "Investigate Acme Corp",
          "hypotheses": ["Review governance and ownership indicators"],
          "tasks": [
            {
              "task_type": "corporate_structure",
              "target_agent": "corporate_agent",
              "description": "Review corporate structure",
              "candidate_tools": ["unknown_tool", "sec_edgar"],
              "priority": "high",
              "rationale": "Ownership and filings are useful"
            }
          ],
          "success_criteria": ["Corporate lane completed"],
          "max_rounds": 1
        }
        """,
    )
    assert plan.tasks[0].candidate_tools == ("sec_edgar",)


def test_plan_investigation_normalizes_noncanonical_target_agent_from_tool_signal():
    plan = plan_investigation(
        "Investigate The Boeing Company for money laundering",
        llm_client=lambda _prompt: """
        {
          "investigation_goal": "Investigate The Boeing Company",
          "hypotheses": ["Potential legal exposure may exist"],
          "tasks": [
            {
              "task_type": "litigation_and_sanctions_screen",
              "target_agent": "The Boeing Company",
              "description": "Screen sanctions and dockets",
              "candidate_tools": ["ofac", "courtlistener"],
              "priority": "high",
              "rationale": "Legal lane should cover sanctions and litigation"
            }
          ],
          "success_criteria": ["Legal lane completed"],
          "max_rounds": 1
        }
        """,
    )
    assert plan.tasks[0].target_agent == "legal_agent"
    assert plan.tasks[0].candidate_tools == ("ofac", "courtlistener")


def test_plan_investigation_accepts_trailing_commas_in_json():
    plan = plan_investigation(
        "Investigate Acme Corp",
        llm_client=lambda _prompt: """
        {
          "investigation_goal": "Investigate Acme Corp",
          "hypotheses": ["Legal exposure possible",],
          "tasks": [
            {
              "task_type": "litigation",
              "target_agent": "legal_agent",
              "description": "Review dockets",
              "candidate_tools": ["courtlistener",],
              "priority": "high",
              "rationale": "Court records can show legal risk",
            },
          ],
          "success_criteria": ["Legal lane completed",],
          "max_rounds": 1,
        }
        """,
    )
    assert plan.tasks[0].target_agent == "legal_agent"
    assert plan.tasks[0].candidate_tools == ("courtlistener",)


def test_plan_investigation_accepts_python_literal_style_payload():
    plan = plan_investigation(
        "Investigate Acme Corp",
        llm_client=lambda _prompt: """{
            'investigation_goal': 'Investigate Acme Corp',
            'hypotheses': ['Governance risk may exist'],
            'tasks': [{
                'task_type': 'corporate_structure',
                'target_agent': 'corporate_agent',
                'description': 'Review corporate filings',
                'candidate_tools': ['sec_edgar'],
                'priority': 'high',
                'rationale': 'SEC filings provide governance signal'
            }],
            'success_criteria': ['Corporate lane completed'],
            'max_rounds': 1
        }""",
    )
    assert plan.tasks[0].target_agent == "corporate_agent"
    assert plan.tasks[0].candidate_tools == ("sec_edgar",)
