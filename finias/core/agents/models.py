from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


class ConfidenceLevel(str, Enum):
    """How confident an agent is in its assessment."""
    VERY_HIGH = "very_high"     # >90% — strong signal, high data quality, multiple confirmations
    HIGH = "high"               # 70-90% — good signal, solid data
    MODERATE = "moderate"       # 50-70% — mixed signals or limited data
    LOW = "low"                 # 30-50% — weak signal, poor data quality, or conflicting indicators
    VERY_LOW = "very_low"       # <30% — speculative, minimal supporting evidence


class SignalDirection(str, Enum):
    """Directional bias of a signal or opinion."""
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"


class MarketRegime(str, Enum):
    """Current market regime classification."""
    RISK_ON = "risk_on"             # Broad risk appetite, expansion
    RISK_OFF = "risk_off"           # Flight to safety, contraction
    TRANSITION = "transition"       # Regime change in progress
    CRISIS = "crisis"               # Extreme stress, dislocations
    LOW_VOLATILITY = "low_vol"      # Complacency, compressed vol
    HIGH_VOLATILITY = "high_vol"    # Elevated vol, uncertainty


class AgentLayer(int, Enum):
    """Hierarchy layer for agent classification."""
    OPERATIONS = 0      # Pure Python computation
    DOMAIN_EXPERT = 1   # Python + Claude interpretation
    DECISION_MAKER = 2  # Claude synthesis of expert opinions
    DIRECTOR = 3        # User-facing conversational interface


class AgentOpinion(BaseModel):
    """
    The fundamental unit of agent communication.

    Every agent produces opinions in this format. Opinions flow UP the hierarchy.
    Higher-layer agents consume opinions from lower-layer agents and synthesize
    them into their own opinions or decisions.
    """
    opinion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str                          # Which agent produced this
    agent_layer: AgentLayer                  # What level in the hierarchy
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # The assessment
    direction: SignalDirection                # Directional bias
    confidence: ConfidenceLevel              # How confident
    regime: Optional[MarketRegime] = None    # Market regime (if applicable)

    # The substance
    summary: str                             # One-paragraph human-readable summary
    key_findings: list[str]                  # Bullet points of most important findings
    data_points: dict[str, Any]              # Raw data backing the opinion

    # Context
    methodology: str                         # How this opinion was formed
    risks_to_view: list[str]                 # What would invalidate this opinion
    watch_items: list[str]                   # What to monitor going forward

    # Metadata
    data_freshness: datetime                 # When the underlying data was last updated
    computation_time_ms: float = 0.0         # How long the computation took

    model_config = {"json_schema_extra": {"examples": []}}


class AgentQuery(BaseModel):
    """
    A structured question posed to an agent.

    Higher-layer agents create queries to ask lower-layer agents for opinions.
    The query format is standardized so any agent can receive and respond to it.
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asking_agent: str           # Who is asking
    target_agent: str           # Who should answer
    question: str               # Natural language question
    context: dict[str, Any] = Field(default_factory=dict)   # Additional context
    require_fresh_data: bool = False   # Force re-computation vs cached
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthStatus(BaseModel):
    """Agent health check response."""
    agent_name: str
    is_healthy: bool
    last_computation: Optional[datetime] = None
    data_freshness: Optional[datetime] = None
    error_message: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class CyclePhase(str, Enum):
    """Business cycle phase classification."""
    EARLY_CYCLE = "early_cycle"       # Recovery — favor cyclicals, small caps
    MID_CYCLE = "mid_cycle"           # Expansion — favor broad market, tech
    LATE_CYCLE = "late_cycle"         # Overheating — favor quality, defensives
    RECESSION = "recession"           # Contraction — favor treasuries, cash, gold


class InflationRegime(str, Enum):
    """Inflation environment classification."""
    DEFLATION_RISK = "deflation_risk"
    DISINFLATION = "disinflation"
    STABLE = "stable"
    RISING = "rising"
    STAGFLATION = "stagflation"


class LiquidityRegime(str, Enum):
    """Liquidity conditions classification."""
    AMPLE = "ample"
    ADEQUATE = "adequate"
    TIGHTENING = "tightening"
    SCARCE = "scarce"


class PolicyStance(str, Enum):
    """Monetary policy stance."""
    EMERGENCY = "emergency"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    HAWKISH = "hawkish"
