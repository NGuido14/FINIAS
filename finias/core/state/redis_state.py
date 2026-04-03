from __future__ import annotations
from typing import Any, Optional
from datetime import datetime, timezone, timedelta
import json
import logging
import redis.asyncio as redis

from finias.core.config.settings import get_settings

logger = logging.getLogger("finias.state")


class RedisState:
    """
    Redis-backed shared state for inter-agent communication.

    Agents publish their latest assessments under standardized keys.
    Other agents can read the latest state without direct coupling.

    Key patterns:
        agent:{name}:latest          — Latest opinion from an agent (JSON)
        agent:{name}:health          — Latest health status (JSON)
        agent:{name}:last_updated    — Timestamp of last update
        regime:current               — Current market regime assessment
        prices:live                  — Live market prices from yfinance (shared, any agent reads)
        data:freshness:{source}      — When data from a source was last refreshed

    All values have TTL to prevent stale data from persisting.
    Default TTL: 1 hour for opinions, 5 minutes for health status.
    """

    DEFAULT_OPINION_TTL = 50400      # 14 hours — covers full trading day from 6:30 AM refresh
    DEFAULT_HEALTH_TTL = 300         # 5 minutes
    DEFAULT_DATA_TTL = 7200          # 2 hours

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    async def initialize(self) -> None:
        """Connect to Redis."""
        settings = get_settings()
        self._client = redis.from_url(
            settings.redis_url,
            decode_responses=True
        )
        # Test connection
        await self._client.ping()
        logger.info("Redis state initialized")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis state closed")

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("Redis state not initialized. Call initialize() first.")
        return self._client

    # --- Agent State ---

    async def publish_opinion(self, agent_name: str, opinion_dict: dict[str, Any]) -> None:
        """Publish an agent's latest opinion to shared state."""
        key = f"agent:{agent_name}:latest"
        opinion_dict["_published_at"] = datetime.now(timezone.utc).isoformat()
        await self.client.setex(key, self.DEFAULT_OPINION_TTL, json.dumps(opinion_dict, default=str))
        await self.client.set(
            f"agent:{agent_name}:last_updated",
            datetime.now(timezone.utc).isoformat()
        )
        logger.debug(f"Published opinion for {agent_name}")

    async def get_latest_opinion(self, agent_name: str) -> Optional[dict[str, Any]]:
        """Get an agent's most recent opinion."""
        key = f"agent:{agent_name}:latest"
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    async def publish_health(self, agent_name: str, health_dict: dict[str, Any]) -> None:
        """Publish an agent's health status."""
        key = f"agent:{agent_name}:health"
        await self.client.setex(key, self.DEFAULT_HEALTH_TTL, json.dumps(health_dict, default=str))

    async def get_health(self, agent_name: str) -> Optional[dict[str, Any]]:
        """Get an agent's latest health status."""
        key = f"agent:{agent_name}:health"
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    # --- Regime State ---

    async def set_regime(self, regime_data: dict[str, Any]) -> None:
        """Update the current market regime assessment."""
        regime_data["_updated_at"] = datetime.now(timezone.utc).isoformat()
        await self.client.setex(
            "regime:current",
            self.DEFAULT_OPINION_TTL,
            json.dumps(regime_data, default=str)
        )

    async def get_regime(self) -> Optional[dict[str, Any]]:
        """Get the current regime assessment."""
        data = await self.client.get("regime:current")
        if data:
            return json.loads(data)
        return None

    # --- Data Freshness ---

    async def mark_data_fresh(self, source: str) -> None:
        """Record that data from a source was just refreshed."""
        await self.client.setex(
            f"data:freshness:{source}",
            self.DEFAULT_DATA_TTL,
            datetime.now(timezone.utc).isoformat()
        )

    async def get_data_freshness(self, source: str) -> Optional[datetime]:
        """When was data from this source last refreshed?"""
        data = await self.client.get(f"data:freshness:{source}")
        if data:
            return datetime.fromisoformat(data)
        return None

    async def is_data_stale(self, source: str, max_age: timedelta = timedelta(hours=1)) -> bool:
        """Check if data from a source is older than max_age."""
        freshness = await self.get_data_freshness(source)
        if freshness is None:
            return True  # No record = stale
        return (datetime.now(timezone.utc) - freshness) > max_age


# Singleton
_state = None

async def get_state() -> RedisState:
    global _state
    if _state is None:
        _state = RedisState()
        await _state.initialize()
    return _state
