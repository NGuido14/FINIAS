from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import logging
import time

from finias.core.agents.models import (
    AgentOpinion, AgentQuery, AgentLayer, HealthStatus
)


class BaseAgent(ABC):
    """
    Abstract base class for all FINIAS agents.

    Every agent in the system inherits from this class.
    It enforces a standard interface:
      - query(): Ask the agent a question, get a structured opinion
      - health_check(): Is the agent operational?
      - Metadata: name, layer, description, capabilities

    The hierarchy:
      Layer 0 (Operations):    Pure Python, no Claude. Fast, free, deterministic.
      Layer 1 (Domain Expert): Python computation + Claude interpretation. Specialized.
      Layer 2 (Decision Maker): Claude synthesis of multiple expert opinions.
      Layer 3 (Director):      User-facing. Routes queries, synthesizes responses.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"finias.agent.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent."""
        ...

    @property
    @abstractmethod
    def layer(self) -> AgentLayer:
        """Which hierarchy layer this agent belongs to."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this agent does."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> list[str]:
        """List of specific things this agent can do / answer."""
        ...

    @abstractmethod
    async def query(self, query: AgentQuery) -> AgentOpinion:
        """
        Ask this agent a question. Returns a structured opinion.

        This is the primary interface. Higher-layer agents call this method
        on lower-layer agents to gather opinions for synthesis.

        Args:
            query: Structured question with context

        Returns:
            AgentOpinion with the agent's assessment
        """
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if this agent is operational and its data is fresh."""
        ...

    async def timed_query(self, query: AgentQuery) -> AgentOpinion:
        """Execute a query and record computation time."""
        start = time.perf_counter()
        opinion = await self.query(query)
        elapsed = (time.perf_counter() - start) * 1000
        opinion.computation_time_ms = elapsed
        self.logger.info(
            f"Query completed in {elapsed:.1f}ms | "
            f"confidence={opinion.confidence.value} | "
            f"direction={opinion.direction.value}"
        )
        return opinion

    def get_tool_definition(self) -> dict[str, Any]:
        """
        Return the Claude tool_use definition for this agent.

        This allows higher-layer agents to call this agent as a tool
        through the Claude API's tool_use feature.
        """
        return {
            "name": f"query_{self.name}",
            "description": (
                f"{self.description}\n\n"
                f"Capabilities:\n" +
                "\n".join(f"- {cap}" for cap in self.capabilities)
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask this agent"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context (e.g., specific ticker, time range)",
                        "default": {}
                    },
                    "require_fresh_data": {
                        "type": "boolean",
                        "description": "Force fresh computation instead of using cached results",
                        "default": False
                    }
                },
                "required": ["question"]
            }
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} layer={self.layer.name}>"
