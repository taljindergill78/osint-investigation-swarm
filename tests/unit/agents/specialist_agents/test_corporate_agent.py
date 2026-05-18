"""Tests for Corporate Agent."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.lead_agent.context_manager import InvestigationContext
from agents.lead_agent.task_planner import SubTask
from agents.specialist_agents.corporate_agent import CorporateAgent
from agents.specialist_agents.corporate_agent.sec_analyzer import summarize_governance_red_flags
from agents.tools import get_available_tools_by_agent, get_tools_for_agent
from osint_swarm.entities import Entity, Evidence


def test_corporate_agent_agent_id():
    agent = CorporateAgent()
    assert agent.agent_id == "corporate_agent"


def test_corporate_agent_ignores_unavailable_selected_tool(monkeypatch):
    """If policy selects an unavailable tool, agent exits without findings."""
    monkeypatch.setattr(
        "agents.specialist_agents.corporate_agent.agent.choose_next_tool",
        lambda **_kwargs: {
            "selected_tool": "unknown_tool",
            "alternatives": [],
            "policy_used": "llm_action_policy",
            "reasoning": "Use ownership tool.",
        },
    )
    agent = CorporateAgent()
    entity = Entity(entity_id="e1", name="E", identifiers={})
    task = SubTask("corporate_structure", "corporate_agent", "Analyze structure")
    ctx = InvestigationContext()
    findings = agent.run(entity, task, ctx)
    assert findings == []


def test_corporate_agent_sec_task_uses_mcp_when_cache_exists():
    data_root = Path("data")
    if not (data_root / "raw" / "sec").exists():
        pytest.skip("no SEC cache")
    agent = CorporateAgent(data_root=data_root)
    entity = Entity(
        entity_id="tesla_inc_cik_0001318605",
        name="Tesla, Inc.",
        identifiers={"cik": "0001318605"},
    )
    task = SubTask("corporate_structure", "corporate_agent", "Analyze structure")
    ctx = InvestigationContext()
    with pytest.MonkeyPatch.context() as mp:
        def _choose_tool(**kwargs):
            used = kwargs.get("used_tools", [])
            if "sec_edgar" in used:
                return {
                    "selected_tool": None,
                    "alternatives": [],
                    "policy_used": "llm_action_policy",
                    "reasoning": "No additional tool required.",
                }
            return {
                "selected_tool": "sec_edgar",
                "alternatives": [],
                "policy_used": "llm_action_policy",
                "reasoning": "Use SEC first.",
            }

        mp.setattr(
            "agents.specialist_agents.corporate_agent.agent.choose_next_tool",
            _choose_tool,
        )
        findings = agent.run(entity, task, ctx)
    assert len(findings) >= 1
    # Should include raw evidence + one summary
    summary_ev = [f for f in findings if "summary" in f.evidence_id or "corporate_summary" in f.evidence_id]
    assert len(summary_ev) >= 1
    assert "SEC" in summary_ev[0].summary or "filing" in summary_ev[0].summary.lower()


def test_summarize_governance_red_flags_empty_returns_empty():
    assert summarize_governance_red_flags([], "e1") == []


def test_summarize_governance_red_flags_adds_summary():
    evidence = [
        Evidence("ev1", "e1", "2024-01-01", "sec_filing", "governance", "8-K filed", "https://sec.gov", confidence=0.9, attributes={"form": "8-K"}),
        Evidence("ev2", "e1", "2024-01-02", "sec_filing", "governance", "10-K filed", "https://sec.gov", confidence=0.95, attributes={"form": "10-K"}),
    ]
    out = summarize_governance_red_flags(evidence, "e1")
    assert len(out) == 1
    assert out[0].evidence_id == "e1_corporate_summary"
    assert out[0].attributes.get("sec_count") == 2
    assert out[0].attributes.get("eight_k_count") == 1


def test_corporate_agent_uses_candidate_tool_order(monkeypatch):
    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, entity, task, context):
            calls.append(self.name)
            return SimpleNamespace(
                tool_name=self.name,
                evidence=[],
                observation=f"{self.name} called",
                discovered_entities=[],
                success=True,
                metadata={},
            )

    monkeypatch.setattr(
        "agents.specialist_agents.corporate_agent.agent.get_tools_for_agent",
        lambda _agent_id, data_root=None, entity=None: {
            "sec_edgar": FakeTool("sec_edgar"),
        },
    )
    monkeypatch.setattr(
        "agents.specialist_agents.corporate_agent.agent.choose_next_tool",
        lambda **_kwargs: {
            "selected_tool": "sec_edgar",
            "alternatives": [],
            "policy_used": "llm_action_policy",
            "reasoning": "Follow candidate tool order.",
        },
    )

    agent = CorporateAgent()
    entity = Entity(entity_id="e1", name="E", identifiers={})
    task = SubTask(
        "corporate_structure",
        "corporate_agent",
        "Analyze structure",
        candidate_tools=("sec_edgar",),
    )
    ctx = InvestigationContext()
    agent.run(entity, task, ctx)

    assert calls[0] == "sec_edgar"


def test_tool_registry_exposes_only_sec_for_corporate(tmp_path):
    entity = Entity(entity_id="e1", name="Example Corp", identifiers={})
    available = get_available_tools_by_agent(data_root=tmp_path, entity=entity)
    corporate_tools = get_tools_for_agent("corporate_agent", data_root=tmp_path, entity=entity)
    assert available["corporate_agent"] == ["sec_edgar"]
    assert list(corporate_tools.keys()) == ["sec_edgar"]


def test_tool_registry_includes_opencorporates_with_token(tmp_path, monkeypatch):
    entity = Entity(
        entity_id="tesla_inc_cik_0001318605",
        name="Tesla, Inc.",
        entity_type="public_company",
        identifiers={"cik": "0001318605"},
    )
    monkeypatch.setenv("OPENCORPORATES_API_TOKEN", "test-token")
    available = get_available_tools_by_agent(data_root=tmp_path, entity=entity)
    corporate_tools = get_tools_for_agent("corporate_agent", data_root=tmp_path, entity=entity)
    assert "opencorporates" in available["corporate_agent"]
    assert "opencorporates" in corporate_tools


def test_tool_registry_includes_opencorporates_with_cache_only(tmp_path, monkeypatch):
    entity = Entity(
        entity_id="tesla_inc_cik_0001318605",
        name="Tesla, Inc.",
        entity_type="public_company",
        identifiers={"cik": "0001318605"},
    )
    monkeypatch.delenv("OPENCORPORATES_API_TOKEN", raising=False)
    cache_dir = tmp_path / "raw" / "opencorporates"
    cache_dir.mkdir(parents=True)
    (cache_dir / "oc_tesla.json").write_text("{}", encoding="utf-8")

    available = get_available_tools_by_agent(data_root=tmp_path, entity=entity)
    corporate_tools = get_tools_for_agent("corporate_agent", data_root=tmp_path, entity=entity)
    assert "opencorporates" in available["corporate_agent"]
    assert "opencorporates" in corporate_tools
