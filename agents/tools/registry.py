"""Default bounded tool registry for agentic specialist agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agents.tools.base import ToolCallResult
from agents.lead_agent.context_manager import InvestigationContext
from agents.lead_agent.task_planner.types import SubTask
from osint_swarm.entities import Entity, Evidence


def _extract_discovered_entities(evidence: List[Evidence], *, source: str) -> List[dict]:
    discovered: List[dict] = []
    for item in evidence:
        attrs = item.attributes or {}
        if attrs.get("officer_name"):
            discovered.append(
                {
                    "name": attrs["officer_name"],
                    "relationship": "officer",
                    "source": source,
                    "identifiers": {},
                    "metadata": {"evidence_id": item.evidence_id, "position": attrs.get("position", "")},
                }
            )
        if attrs.get("controlling_entity_name"):
            discovered.append(
                {
                    "name": attrs["controlling_entity_name"],
                    "relationship": "controlling_entity",
                    "source": source,
                    "identifiers": {},
                    "metadata": {"evidence_id": item.evidence_id},
                }
            )
        if attrs.get("ubo_name"):
            discovered.append(
                {
                    "name": attrs["ubo_name"],
                    "relationship": "ultimate_beneficial_owner",
                    "source": source,
                    "identifiers": {},
                    "metadata": {"evidence_id": item.evidence_id},
                }
            )
    return discovered


class _BaseTool:
    name = ""
    description = ""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else Path("data")


class SecEdgarTool(_BaseTool):
    name = "sec_edgar"
    description = "Fetch SEC EDGAR evidence and governance red flags."

    def run(self, entity: Entity, task: SubTask, context: InvestigationContext) -> ToolCallResult:
        from mcp_layer import get_evidence_for_entity
        from agents.specialist_agents.corporate_agent.sec_analyzer.analyzer import summarize_governance_red_flags

        evidence = get_evidence_for_entity(entity, sources=("sec_edgar",), data_root=self.data_root)
        if task.task_type in {"corporate_structure", "sec_filings", "transaction_patterns"}:
            evidence = evidence + summarize_governance_red_flags(evidence, entity.entity_id)
        observation = f"Retrieved {len(evidence)} SEC/governance evidence rows."
        return ToolCallResult(tool_name=self.name, evidence=evidence, observation=observation)


class OfacTool(_BaseTool):
    name = "ofac"
    description = "Run OFAC sanctions screening against cached SDN data."

    def run(self, entity: Entity, task: SubTask, context: InvestigationContext) -> ToolCallResult:
        from agents.specialist_agents.legal_agent.sanctions_screener.screener import screen as ofac_screen

        evidence = ofac_screen(entity, task, context, data_root=self.data_root)
        screened = any(item.attributes.get("screened") for item in evidence)
        observation = "Completed OFAC screening." if screened else "Completed OFAC screening (no matches)."
        return ToolCallResult(
            tool_name=self.name,
            evidence=evidence,
            observation=observation,
            success=True,
        )


class CourtListenerTool(_BaseTool):
    name = "courtlistener"
    description = "Fetch litigation and regulatory court records from CourtListener."

    def run(self, entity: Entity, task: SubTask, context: InvestigationContext) -> ToolCallResult:
        from agents.specialist_agents.legal_agent.pacer_analyzer.analyzer import fetch as court_fetch

        evidence = court_fetch(entity, task, context, data_root=self.data_root)
        observation = f"Retrieved {len(evidence)} CourtListener evidence rows."
        return ToolCallResult(
            tool_name=self.name,
            evidence=evidence,
            observation=observation,
            success=True,
        )


class OpenCorporatesTool(_BaseTool):
    name = "opencorporates"
    description = "Map beneficial ownership and control relationships."

    def run(self, entity: Entity, task: SubTask, context: InvestigationContext) -> ToolCallResult:
        from agents.specialist_agents.corporate_agent.structure_mapper.mapper import map_structure

        evidence = map_structure(entity, task, context, data_root=self.data_root)
        discovered = _extract_discovered_entities(evidence, source=self.name)
        observation = f"Retrieved {len(evidence)} OpenCorporates evidence rows."
        return ToolCallResult(
            tool_name=self.name,
            evidence=evidence,
            observation=observation,
            discovered_entities=discovered,
            success=True,
        )


class GdeltTool(_BaseTool):
    name = "gdelt"
    description = "Fetch adverse media and public reporting from GDELT."

    def run(self, entity: Entity, task: SubTask, context: InvestigationContext) -> ToolCallResult:
        from mcp_layer import get_evidence_for_entity

        evidence = get_evidence_for_entity(entity, sources=("gdelt",), data_root=self.data_root)
        relevant = sum(1 for item in evidence if item.attributes.get("relevant"))
        observation = f"Retrieved {len(evidence)} GDELT articles ({relevant} relevant)."
        return ToolCallResult(tool_name=self.name, evidence=evidence, observation=observation)


def get_tools_for_agent(
    agent_id: str,
    data_root: Optional[Path] = None,
    *,
    entity: Optional[Entity] = None,
) -> Dict[str, _BaseTool]:
    """Return the bounded toolset available to a specialist agent."""
    corporate_tools: Dict[str, _BaseTool] = {"sec_edgar": SecEdgarTool(data_root=data_root)}
    if entity is not None and str(getattr(entity, "entity_type", "") or "").lower() == "public_company":
        corporate_tools["opencorporates"] = OpenCorporatesTool(data_root=data_root)
    toolsets: Dict[str, Dict[str, _BaseTool]] = {
        "corporate_agent": corporate_tools,
        "legal_agent": {
            "ofac": OfacTool(data_root=data_root),
            "courtlistener": CourtListenerTool(data_root=data_root),
        },
        "social_graph_agent": {
            "gdelt": GdeltTool(data_root=data_root),
        },
    }
    return toolsets.get(agent_id, {})


def get_available_tools_by_agent(
    data_root: Optional[Path] = None,
    *,
    entity: Optional[Entity] = None,
) -> Dict[str, List[str]]:
    """Return a planner-friendly map of available tool names per agent."""
    corporate_tools = ["sec_edgar"]
    if entity is not None and str(getattr(entity, "entity_type", "") or "").lower() == "public_company":
        corporate_tools.append("opencorporates")
    return {
        "corporate_agent": corporate_tools,
        "legal_agent": ["ofac", "courtlistener"],
        "social_graph_agent": ["gdelt"],
    }
