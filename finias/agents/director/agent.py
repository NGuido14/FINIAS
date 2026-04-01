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
from finias.core.state.redis_state import RedisState
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("finias.agent.director")


class Director(BaseAgent):
    """
    The Director — your interface to FINIAS.

    Receives natural language queries, routes to specialist agents via
    Claude tool_use, and synthesizes grounded responses.
    """

    def __init__(self, registry: ToolRegistry, state: Optional[RedisState] = None):
        super().__init__()
        self.registry = registry
        self.state = state
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

    async def _get_cached_macro_context(self) -> Optional[str]:
        """
        Check Redis for a fresh macro regime assessment.

        Returns a formatted context string if cache is fresh (< 12 hours),
        or None if stale/missing. When a fresh cache is available, the
        Director can answer macro questions directly without calling the
        macro agent tool — saving ~$0.30 and 5-15 seconds per query.
        """
        if self.state is None:
            return None

        try:
            regime = await self.state.get_regime()
            if not regime:
                return None

            # Check freshness
            updated_at = regime.get("_updated_at")
            if not updated_at:
                return None

            updated_time = datetime.fromisoformat(updated_at)
            age = datetime.now(timezone.utc) - updated_time
            if age > timedelta(hours=12):
                logger.debug(f"Cached regime is {age.total_seconds()/3600:.1f}h old — stale")
                return None

            # Build a concise context summary from the cached regime
            # This is what the Director uses instead of calling the macro agent
            interp = regime.get("interpretation", {})
            trajectory = regime.get("trajectory", {})
            key_levels = regime.get("key_levels", {})
            regime_info = regime.get("regime", {})
            category_scores = regime.get("category_scores", {})

            parts = []
            parts.append(f"CACHED MACRO CONTEXT (updated {age.total_seconds()/60:.0f} minutes ago):")
            parts.append("")

            # Regime classification
            parts.append(f"Regime: {regime_info.get('primary', 'unknown')} | "
                        f"Cycle: {regime_info.get('cycle_phase', 'unknown')} | "
                        f"Volatility: {regime_info.get('volatility', 'unknown')} | "
                        f"Inflation: {regime_info.get('inflation', 'unknown')} | "
                        f"Liquidity: {regime_info.get('liquidity', 'unknown')}")

            # Composite and binding
            parts.append(f"Composite: {regime.get('composite_score', 0):.3f} | "
                        f"Stress: {regime.get('stress_index', 0):.3f} | "
                        f"Confidence: {regime.get('confidence', 0):.2f} | "
                        f"Binding: {regime.get('binding_constraint', 'unknown')}")

            # Category scores
            parts.append(f"Scores — Growth: {category_scores.get('growth_cycle', 0):+.3f}, "
                        f"Monetary: {category_scores.get('monetary_liquidity', 0):+.3f}, "
                        f"Inflation: {category_scores.get('inflation', 0):+.3f}, "
                        f"Market: {category_scores.get('market_signals', 0):+.3f}")

            # Key levels
            kl = key_levels
            parts.append(f"Key Levels — VIX: {kl.get('vix', 'N/A')}, "
                        f"Fed Funds: {kl.get('fed_funds', 'N/A')}%, "
                        f"Core PCE: {kl.get('core_pce_yoy', 'N/A'):.2f}%, "
                        f"HY Spread: {kl.get('hy_spread', 'N/A')}%, "
                        f"Sahm: {kl.get('sahm_value', 'N/A')}, "
                        f"Net Liq: ${kl.get('net_liquidity', 0)/1_000_000:.2f}T")

            # Trajectory
            forward_bias = trajectory.get("forward_bias", {})
            sizing = trajectory.get("position_sizing", {})
            velocity = trajectory.get("velocity", {})
            events = trajectory.get("event_calendar", {})

            parts.append(f"Forward Bias: {forward_bias.get('bias', 'unknown')} "
                        f"(score: {forward_bias.get('score', 0):+.3f}, "
                        f"confidence: {forward_bias.get('confidence', 'unknown')})")

            parts.append(f"Position Sizing — Max: {sizing.get('max_single_position_pct', 'N/A')}%, "
                        f"Beta: {sizing.get('portfolio_beta_target', 'N/A')}, "
                        f"Cash: {sizing.get('cash_target_pct', 'N/A')}%, "
                        f"Reduce Exposure: {sizing.get('reduce_overall_exposure', False)}")

            parts.append(f"Velocity — VIX: {velocity.get('vix', 'unknown')}, "
                        f"Spreads: {velocity.get('credit_spreads', 'unknown')}, "
                        f"Breadth: {velocity.get('breadth', 'unknown')}, "
                        f"Urgency: {velocity.get('urgency', 'unknown')}")

            # Upcoming events
            upcoming = events.get("upcoming_events", [])
            if upcoming:
                event_strs = [f"{e['event']} in {e['days_away']}d" for e in upcoming[:3]]
                parts.append(f"Events: {', '.join(event_strs)} | "
                            f"Sizing multiplier: {events.get('pre_event_sizing_multiplier', 1.0)}x")

            # Scenario triggers (closest ones)
            triggers = trajectory.get("scenario_triggers", [])
            close_triggers = sorted(
                [t for t in triggers if isinstance(t.get("distance"), (int, float))],
                key=lambda t: abs(t["distance"])
            )[:3]
            if close_triggers:
                trig_strs = [f"{t['id']}: {t['current']} → {t['threshold']} (dist: {t['distance']:.2f})"
                            for t in close_triggers]
                parts.append(f"Nearest Triggers: {'; '.join(trig_strs)}")

            # Interpretation summary and key findings
            if interp.get("summary"):
                parts.append(f"\nInterpretation Summary: {interp['summary']}")
            if interp.get("key_findings"):
                parts.append("Key Findings:")
                for i, finding in enumerate(interp["key_findings"], 1):
                    parts.append(f"  {i}. {finding}")
            if interp.get("risks"):
                parts.append("Risks:")
                for risk in interp["risks"]:
                    parts.append(f"  - {risk}")
            if interp.get("watch_items"):
                parts.append("Watch Items:")
                for item in interp["watch_items"]:
                    parts.append(f"  - {item}")

            # Key metrics
            metrics = interp.get("key_metrics", {})
            if metrics:
                parts.append(f"\nKey Metrics: {json.dumps(metrics)}")

            return "\n".join(parts)

        except Exception as e:
            logger.warning(f"Failed to load cached macro context: {e}")
            return None

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

        # Check for cached macro context to avoid expensive macro agent calls
        cached_context = await self._get_cached_macro_context()
        if cached_context:
            dated_system_prompt += (
                "\n\nYou have access to a FRESH cached macro regime assessment below. "
                "Use this data to answer macro-related questions directly WITHOUT calling "
                "the macro strategist tool. The data is current and comprehensive. "
                "Only call the macro strategist tool if the user explicitly asks for a "
                "fresh/new assessment or if the question requires analysis beyond what's "
                "in the cached context.\n\n" + cached_context
            )
            logger.info("Using cached macro context (skipping macro agent call)")

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
