"""
Agent Template

Copy this directory to create a new agent.
Replace all placeholders with your agent's specifics.
"""

from __future__ import annotations
from typing import Any
from datetime import datetime, timezone

from finias.core.agents.base import BaseAgent
from finias.core.agents.models import (
    AgentOpinion, AgentQuery, AgentLayer,
    ConfidenceLevel, SignalDirection, HealthStatus
)


class TemplateAgent(BaseAgent):
    """Template agent — copy and customize."""

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "template"

    @property
    def layer(self) -> AgentLayer:
        return AgentLayer.DOMAIN_EXPERT

    @property
    def description(self) -> str:
        return "Template agent — replace with your agent's description."

    @property
    def capabilities(self) -> list[str]:
        return [
            "Capability 1",
            "Capability 2",
        ]

    async def query(self, query: AgentQuery) -> AgentOpinion:
        """Process a query and return an opinion."""
        return AgentOpinion(
            agent_name=self.name,
            agent_layer=self.layer,
            direction=SignalDirection.NEUTRAL,
            confidence=ConfidenceLevel.MODERATE,
            summary="Template response — replace with real logic.",
            key_findings=["Finding 1"],
            data_points={},
            methodology="Template methodology",
            risks_to_view=["Risk 1"],
            watch_items=["Watch item 1"],
            data_freshness=datetime.now(timezone.utc),
        )

    async def health_check(self) -> HealthStatus:
        """Check agent health."""
        return HealthStatus(
            agent_name=self.name,
            is_healthy=True,
        )
