from __future__ import annotations
from typing import Any, Optional
import logging

from finias.core.agents.base import BaseAgent
from finias.core.agents.models import AgentQuery, AgentOpinion, AgentLayer


logger = logging.getLogger("finias.registry")


class ToolRegistry:
    """
    Central registry for all FINIAS agents.

    Agents register themselves here. Higher-layer agents query the registry
    to discover available tools and invoke them. This keeps layers decoupled:
    the Director doesn't import the Macro Strategist — it asks the registry.

    Usage:
        registry = ToolRegistry()
        registry.register(macro_agent)

        # Get tool definitions for Claude API
        tools = registry.get_tool_definitions()

        # Handle a tool call from Claude
        result = await registry.handle_tool_call("query_macro_strategist", {...})
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent as an available tool."""
        tool_name = f"query_{agent.name}"
        if tool_name in self._agents:
            logger.warning(f"Overwriting existing agent registration: {tool_name}")
        self._agents[tool_name] = agent
        logger.info(f"Registered agent: {agent.name} (layer={agent.layer.name})")

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the registry."""
        tool_name = f"query_{agent_name}"
        if tool_name in self._agents:
            del self._agents[tool_name]
            logger.info(f"Unregistered agent: {agent_name}")

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(f"query_{agent_name}")

    def get_tool_definitions(self, max_layer: Optional[AgentLayer] = None) -> list[dict[str, Any]]:
        """
        Get Claude tool_use definitions for all registered agents.

        Args:
            max_layer: Only include agents at or below this layer.
                       E.g., max_layer=DOMAIN_EXPERT excludes decision makers.
        """
        tools = []
        for agent in self._agents.values():
            if max_layer is not None and agent.layer.value > max_layer.value:
                continue
            tools.append(agent.get_tool_definition())
        return tools

    async def handle_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        calling_agent: str = "unknown"
    ) -> AgentOpinion:
        """
        Handle a tool call from a Claude agent.

        This is called when Claude's response includes a tool_use block.
        It routes the call to the correct agent and returns the opinion.

        Args:
            tool_name: The tool name from Claude's response (e.g., "query_macro_strategist")
            tool_input: The input parameters from Claude
            calling_agent: Name of the agent making the call

        Returns:
            AgentOpinion from the queried agent

        Raises:
            KeyError: If the tool_name doesn't match any registered agent
        """
        if tool_name not in self._agents:
            available = list(self._agents.keys())
            raise KeyError(
                f"Unknown tool: {tool_name}. Available tools: {available}"
            )

        agent = self._agents[tool_name]
        # Merge any extra tool parameters (e.g., symbols, metrics) into context
        # so agents can access tool-specific params beyond the standard 3
        context = tool_input.get("context", {})
        for key, value in tool_input.items():
            if key not in ("question", "context", "require_fresh_data"):
                context[key] = value

        query = AgentQuery(
            asking_agent=calling_agent,
            target_agent=agent.name,
            question=tool_input.get("question", ""),
            context=context,
            require_fresh_data=tool_input.get("require_fresh_data", False)
        )

        logger.info(f"{calling_agent} → {agent.name}: {query.question[:100]}...")
        opinion = await agent.timed_query(query)
        logger.info(
            f"{agent.name} → {calling_agent}: "
            f"confidence={opinion.confidence.value}, "
            f"direction={opinion.direction.value}"
        )
        return opinion

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents with their metadata."""
        return [
            {
                "name": agent.name,
                "layer": agent.layer.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "tool_name": f"query_{agent.name}"
            }
            for agent in self._agents.values()
        ]

    @property
    def agent_count(self) -> int:
        return len(self._agents)
