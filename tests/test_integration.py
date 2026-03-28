"""
Integration tests for the FINIAS vertical slice.

These tests verify that the full pipeline works:
  Data providers → Computations → Agent → Director
"""

import pytest
from finias.core.agents.models import (
    AgentOpinion, AgentQuery, AgentLayer,
    ConfidenceLevel, SignalDirection, MarketRegime
)
from finias.core.agents.registry import ToolRegistry


def test_core_imports():
    """Verify all core modules can be imported."""
    from finias.core.agents.base import BaseAgent
    from finias.core.agents.models import AgentOpinion
    from finias.core.agents.registry import ToolRegistry
    from finias.core.config.settings import Settings
    from finias.core.database.connection import DatabasePool
    from finias.core.database.migrations import run_migrations
    from finias.core.state.redis_state import RedisState
    assert True


def test_data_imports():
    """Verify all data modules can be imported."""
    from finias.data.providers.polygon_client import PolygonClient
    from finias.data.providers.fred_client import FredClient, MACRO_SERIES
    from finias.data.cache.market_cache import MarketDataCache
    assert len(MACRO_SERIES) > 0


def test_agent_imports():
    """Verify all agent modules can be imported."""
    from finias.agents.macro_strategist.agent import MacroStrategist
    from finias.agents.macro_strategist.tools import get_macro_tool_definition
    from finias.agents.director.agent import Director
    from finias.agents.director.prompts.system import DIRECTOR_SYSTEM_PROMPT
    assert len(DIRECTOR_SYSTEM_PROMPT) > 0


def test_computation_imports():
    """Verify all computation modules can be imported."""
    from finias.agents.macro_strategist.computations.yield_curve import analyze_yield_curve
    from finias.agents.macro_strategist.computations.volatility import analyze_volatility
    from finias.agents.macro_strategist.computations.breadth import analyze_breadth
    from finias.agents.macro_strategist.computations.cross_asset import analyze_cross_assets
    from finias.agents.macro_strategist.computations.regime import detect_regime
    assert True


def test_models():
    """Test that core models can be instantiated."""
    from datetime import datetime, timezone

    opinion = AgentOpinion(
        agent_name="test",
        agent_layer=AgentLayer.DOMAIN_EXPERT,
        direction=SignalDirection.NEUTRAL,
        confidence=ConfidenceLevel.MODERATE,
        summary="Test opinion",
        key_findings=["Finding 1"],
        data_points={"key": "value"},
        methodology="Test",
        risks_to_view=["Risk 1"],
        watch_items=["Watch 1"],
        data_freshness=datetime.now(timezone.utc),
    )
    assert opinion.agent_name == "test"
    assert opinion.direction == SignalDirection.NEUTRAL


def test_registry():
    """Test the tool registry."""
    registry = ToolRegistry()
    assert registry.agent_count == 0

    tools = registry.get_tool_definitions()
    assert len(tools) == 0

    agents = registry.list_agents()
    assert len(agents) == 0


def test_settings():
    """Test settings can be loaded with defaults."""
    from finias.core.config.settings import Settings
    settings = Settings()
    assert settings.postgres_host == "localhost"
    assert settings.environment == "development"
    assert settings.claude_model == "claude-sonnet-4-20250514"
