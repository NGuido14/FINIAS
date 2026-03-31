"""
Director Agent

Layer 3: The user-facing conversational agent.

The Director receives natural language from the user, decides which
specialist agents to consult, synthesizes their responses, and
communicates back in clear, grounded language.

Implementation uses Claude's tool_use feature:
  1. User message + system prompt + tool definitions → Claude
  2. Claude decides which tools to call (if any)
  3. Tool calls are routed through the ToolRegistry
  4. Tool results are sent back to Claude for synthesis
  5. Claude produces the final response
"""

from __future__ import annotations
from typing import Any, Optional
import json
import logging

import anthropic

from finias.core.agents.base import BaseAgent
from finias.core.agents.models import (
    AgentOpinion, AgentQuery, AgentLayer,
    ConfidenceLevel, SignalDirection, HealthStatus
)
from finias.core.agents.registry import ToolRegistry
from finias.core.config.settings import get_settings
from finias.agents.director.prompts.system import DIRECTOR_SYSTEM_PROMPT

logger = logging.getLogger("finias.agent.director")


class Director(BaseAgent):
    """
    The Director — your interface to FINIAS.

    Receives natural language queries, routes to specialist agents via
    Claude tool_use, and synthesizes grounded responses.
    """

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self.registry = registry
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model_fast
        self._max_tokens = settings.claude_max_tokens
        self._conversation_history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "director"

    @property
    def layer(self) -> AgentLayer:
        return AgentLayer.DIRECTOR

    @property
    def description(self) -> str:
        return "Director: User-facing conversational agent that coordinates specialist agents."

    @property
    def capabilities(self) -> list[str]:
        return [
            "Natural language conversation about markets and portfolio",
            "Route queries to specialist agents",
            "Synthesize multi-agent opinions",
            "Explain decisions and reasoning",
        ]

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main entry point for the CLI interface.
        Manages conversation history and handles the tool_use loop.
        """
        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # Get tool definitions from registry
        tools = self.registry.get_tool_definitions()

        # Prepend today's date so Claude uses correct timeframes
        from datetime import date as _date
        dated_system_prompt = f"TODAY'S DATE: {_date.today().isoformat()}. All dates and timeframes in your response must be relative to today. Do not reference 2024 or 2025 as future dates.\n\n" + DIRECTOR_SYSTEM_PROMPT

        # Initial Claude call
        from finias.core.utils.retry import retry_claude_call

        response = await retry_claude_call(
            lambda: self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=dated_system_prompt,
                tools=tools if tools else None,
                messages=self._conversation_history,
            )
        )

        # Handle tool_use loop
        while response.stop_reason == "tool_use":
            # Extract tool calls
            tool_results = []
            assistant_content = response.content

            for block in assistant_content:
                if block.type == "tool_use":
                    logger.info(f"Director calling tool: {block.name}")
                    try:
                        opinion = await self.registry.handle_tool_call(
                            tool_name=block.name,
                            tool_input=block.input,
                            calling_agent=self.name,
                        )
                        # Convert opinion to string for Claude
                        result_text = json.dumps({
                            "direction": opinion.direction.value,
                            "confidence": opinion.confidence.value,
                            "regime": opinion.regime.value if opinion.regime else None,
                            "summary": opinion.summary,
                            "key_findings": opinion.key_findings,
                            "risks_to_view": opinion.risks_to_view,
                            "watch_items": opinion.watch_items,
                            "data_points": opinion.data_points,
                        }, default=str)
                    except Exception as e:
                        logger.error(f"Tool call failed: {block.name}: {e}")
                        result_text = json.dumps({
                            "error": str(e),
                            "message": f"Agent {block.name} encountered an error."
                        })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            # Add assistant response and tool results to history
            self._conversation_history.append({
                "role": "assistant",
                "content": assistant_content,
            })
            self._conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            # Continue the conversation with tool results
            response = await retry_claude_call(
                lambda: self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=dated_system_prompt,
                    tools=tools if tools else None,
                    messages=self._conversation_history,
                )
            )

        # Extract final text response
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        # Add to history
        self._conversation_history.append({
            "role": "assistant",
            "content": response.content,
        })

        # Trim history if too long (keep last 20 turns)
        if len(self._conversation_history) > 40:
            self._conversation_history = self._conversation_history[-40:]

        return final_text

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []

    async def query(self, query: AgentQuery) -> AgentOpinion:
        """Implement BaseAgent interface (not used directly for Director)."""
        response = await self.chat(query.question)
        return AgentOpinion(
            agent_name=self.name,
            agent_layer=self.layer,
            direction=SignalDirection.NEUTRAL,
            confidence=ConfidenceLevel.MODERATE,
            summary=response,
            key_findings=[],
            data_points={},
            methodology="Director synthesis via Claude tool_use",
            risks_to_view=[],
            watch_items=[],
            data_freshness=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )

    async def health_check(self) -> HealthStatus:
        """Check Director health — mainly that Claude API is reachable."""
        try:
            # Quick API test
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return HealthStatus(
                agent_name=self.name,
                is_healthy=True,
                details={
                    "agents_available": self.registry.agent_count,
                    "conversation_length": len(self._conversation_history),
                }
            )
        except Exception as e:
            return HealthStatus(
                agent_name=self.name,
                is_healthy=False,
                error_message=str(e),
            )
