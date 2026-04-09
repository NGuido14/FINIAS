"""
Technical Analyst Agent

Layer 1 Domain Expert: Price-based technical analysis.

This agent:
  1. Loads OHLCV data from PostgreSQL via batch loading
  2. Converts to pandas DataFrames
  3. Runs pure Python computation modules (trend, momentum, levels)
  4. Returns a structured AgentOpinion
  5. Caches results in Redis for Director fast queries
  6. Persists to PostgreSQL for prediction tracking

All computation is pure Python + pandas-ta. No Claude API calls.
Cost: $0.00 per run.
"""

from __future__ import annotations
from typing import Any, Optional
from datetime import date, datetime, timedelta, timezone
import json
import logging

import pandas as pd

from finias.core.agents.base import BaseAgent
from finias.core.agents.models import (
    AgentOpinion, AgentQuery, AgentLayer,
    ConfidenceLevel, SignalDirection, HealthStatus
)
from finias.core.state.redis_state import RedisState
from finias.data.cache.market_cache import MarketDataCache
from finias.agents.technical_analyst.tools import get_ta_tool_definition

# Computation modules
from finias.agents.technical_analyst.computations.trend import analyze_trend
from finias.agents.technical_analyst.computations.momentum import analyze_momentum
from finias.agents.technical_analyst.computations.levels import analyze_levels
from finias.agents.technical_analyst.computations.volume import analyze_volume
from finias.agents.technical_analyst.computations.relative_strength import (
    analyze_relative_strength,
    compute_universe_returns,
    SECTOR_ETF_MAP,
)

logger = logging.getLogger("finias.agent.technical_analyst")

# Minimum bars needed for full analysis (200-day MA)
MIN_BARS = 200

# How many years of history to load for computation
LOOKBACK_YEARS = 2


class TechnicalAnalyst(BaseAgent):
    """
    The Technical Analyst — reads price action.

    Produces multi-timeframe trend analysis, regime-adaptive momentum signals,
    and key support/resistance levels across the S&P 500 universe.
    All computation is pure Python — free, fast, deterministic.
    """

    def __init__(self, cache: MarketDataCache, state: RedisState):
        super().__init__()
        self.cache = cache
        self.state = state
        self._last_computation: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "technical_analyst"

    @property
    def layer(self) -> AgentLayer:
        return AgentLayer.DOMAIN_EXPERT

    @property
    def description(self) -> str:
        return (
            "Technical Analyst: Analyzes price action across the S&P 500 universe "
            "using multi-timeframe trend analysis (Ichimoku, ADX, MA constellation), "
            "regime-adaptive momentum (RSI, MACD, Stochastic with adaptive thresholds), "
            "and support/resistance level identification with key level clustering. "
            "All computation is pure Python — precise, deterministic, $0.00 per run."
        )

    @property
    def capabilities(self) -> list[str]:
        return [
            "Multi-timeframe trend analysis (Ichimoku Cloud, ADX, MA constellation)",
            "Trend regime classification (strong_uptrend → strong_downtrend)",
            "Regime-adaptive momentum analysis (RSI, MACD, Stochastic)",
            "Momentum divergence detection (regular and hidden)",
            "Support and resistance level identification with clustering",
            "Risk/reward ratio computation from key levels",
            "Sector-level technical aggregation",
            "Scan for stocks matching technical criteria",
        ]

    def get_tool_definition(self) -> dict[str, Any]:
        """Override BaseAgent to use custom tool definition with symbols param."""
        return get_ta_tool_definition()

    async def query(self, query: AgentQuery) -> AgentOpinion:
        """
        Process a technical analysis query.

        Pipeline: Load Data → Convert to DataFrames → Compute → Cache → Return
        """
        start_time = datetime.now(timezone.utc)

        # Determine symbols to analyze
        symbols = self._resolve_symbols(query)

        # Load price data
        from_date = date.today() - timedelta(days=365 * LOOKBACK_YEARS)
        bars_by_symbol = await self.cache.get_batch_daily_bars(symbols, from_date)

        if not bars_by_symbol:
            return self._empty_opinion("No price data available for requested symbols")

        # Convert to DataFrames
        dfs = self._bars_to_dataframes(bars_by_symbol)

        # Load sector context for relative strength analysis
        sector_map = await self._load_sector_map()
        sector_etf_dfs = self._ensure_sector_etfs(dfs, bars_by_symbol)
        spy_df = dfs.get("SPY")

        # Compute universe 20d returns for RS percentile ranking
        universe_returns = compute_universe_returns(dfs)

        # Run computation for each symbol
        all_signals = {}
        skipped = 0

        for symbol, df in dfs.items():
            if len(df) < MIN_BARS:
                skipped += 1
                continue

            try:
                # Sequential: trend → momentum → levels → volume → relative_strength
                trend = analyze_trend(df, symbol=symbol)
                momentum = analyze_momentum(df, symbol=symbol, trend_regime=trend.trend_regime)
                levels = analyze_levels(df, symbol=symbol)
                vol = analyze_volume(df, symbol=symbol, trend_regime=trend.trend_regime)

                # Relative strength needs sector ETF and SPY bars
                sector_name = sector_map.get(symbol, "unknown")
                sector_etf = SECTOR_ETF_MAP.get(sector_name)
                sector_etf_df = sector_etf_dfs.get(sector_etf) if sector_etf else None

                rs = analyze_relative_strength(
                    df, symbol=symbol, sector=sector_name,
                    sector_etf_df=sector_etf_df, spy_df=spy_df,
                    universe_returns_20d=universe_returns,
                )

                all_signals[symbol] = {
                    "trend": trend.to_dict(),
                    "momentum": momentum.to_dict(),
                    "levels": levels.to_dict(),
                    "volume": vol.to_dict(),
                    "relative_strength": rs.to_dict(),
                }
            except Exception as e:
                logger.warning(f"Computation failed for {symbol}: {e}")

        if not all_signals:
            return self._empty_opinion("All symbols had insufficient data for analysis")

        # Build summary statistics
        summary_stats = self._build_summary(all_signals)

        # Cache in Redis
        await self._cache_results(all_signals, summary_stats)

        # Persist to PostgreSQL
        await self._persist_signals(all_signals)

        self._last_computation = datetime.now(timezone.utc)

        # Build and return opinion
        return self._build_opinion(all_signals, summary_stats, query, skipped)

    def _resolve_symbols(self, query: AgentQuery) -> list[str]:
        """Determine which symbols to analyze from the query."""
        # Check for explicit symbols in query context
        symbols = query.context.get("symbols", [])
        if not symbols:
            symbols = query.context.get("tool_input", {}).get("symbols", [])

        if not symbols:
            from finias.data.universe import MACRO_ETFS
            symbols = list(MACRO_ETFS)

        # Always ensure sector ETFs + SPY are loaded for RS computation
        from finias.data.universe import MACRO_ETFS
        essential = set(MACRO_ETFS)  # Includes SPY and all sector ETFs
        combined = list(set(symbols) | essential)
        return combined

    async def _load_sector_map(self) -> dict[str, str]:
        """Load GICS sector mapping from symbol_universe table."""
        try:
            rows = await self.cache.db.fetch(
                "SELECT symbol, sector FROM symbol_universe WHERE is_active AND sector IS NOT NULL"
            )
            return {r["symbol"]: r["sector"] for r in rows}
        except Exception:
            return {}

    def _ensure_sector_etfs(
        self, dfs: dict[str, pd.DataFrame], bars_by_symbol: dict,
    ) -> dict[str, pd.DataFrame]:
        """Ensure all sector ETF DataFrames are available."""
        etf_dfs = {}
        for etf in SECTOR_ETF_MAP.values():
            if etf in dfs:
                etf_dfs[etf] = dfs[etf]
            elif etf in bars_by_symbol:
                # Convert if not already in dfs
                df = pd.DataFrame(bars_by_symbol[etf])
                df = df.rename(columns={"trade_date": "date"})
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.sort_values("date").reset_index(drop=True)
                etf_dfs[etf] = df
        return etf_dfs

    def _bars_to_dataframes(self, bars_by_symbol: dict) -> dict[str, pd.DataFrame]:
        """Convert bar dicts to pandas DataFrames suitable for pandas-ta."""
        dfs = {}
        for symbol, bars in bars_by_symbol.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            # Ensure correct column names for pandas-ta
            df = df.rename(columns={
                "trade_date": "date",
            })
            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            dfs[symbol] = df
        return dfs

    def _build_summary(self, all_signals: dict) -> dict:
        """Build aggregate summary across all analyzed symbols."""
        trends = {"strong_uptrend": 0, "uptrend": 0, "consolidation": 0,
                  "downtrend": 0, "strong_downtrend": 0, "unknown": 0}
        divergences = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for symbol, sig in all_signals.items():
            trend = sig["trend"]
            mom = sig["momentum"]
            regime = trend.get("trend_regime", "unknown")
            trends[regime] = trends.get(regime, 0) + 1

            # Count directional bias
            ts = trend.get("trend_score", 0)
            ms = mom.get("momentum_score", 0)
            combined = (ts + ms) / 2
            if combined > 0.15:
                bullish_count += 1
            elif combined < -0.15:
                bearish_count += 1
            else:
                neutral_count += 1

            # Track divergences
            div = mom.get("divergence", {}).get("type", "none")
            if div != "none":
                divergences.append({"symbol": symbol, "type": div})

        # Volume confirmation stats
        vol_confirming = 0
        vol_contradicting = 0
        rs_improving = 0
        rs_deteriorating = 0

        for symbol, sig in all_signals.items():
            vc = sig.get("volume", {}).get("volume_confirmation_score", 0)
            if vc > 0.2:
                vol_confirming += 1
            elif vc < -0.2:
                vol_contradicting += 1

            rs_r = sig.get("relative_strength", {}).get("rs_regime", "neutral")
            if rs_r in ("leading", "improving"):
                rs_improving += 1
            elif rs_r in ("lagging", "deteriorating"):
                rs_deteriorating += 1

        total = bullish_count + bearish_count + neutral_count
        return {
            "total_analyzed": total,
            "pct_bullish": round(bullish_count / total * 100, 1) if total > 0 else 0,
            "pct_bearish": round(bearish_count / total * 100, 1) if total > 0 else 0,
            "pct_neutral": round(neutral_count / total * 100, 1) if total > 0 else 0,
            "trend_distribution": trends,
            "divergences": divergences[:10],
            "volume_confirming": vol_confirming,
            "volume_contradicting": vol_contradicting,
            "rs_improving": rs_improving,
            "rs_deteriorating": rs_deteriorating,
        }

    async def _cache_results(self, all_signals: dict, summary: dict):
        """Cache computation results in Redis for Director fast queries."""
        if self.state is None:
            return

        try:
            # Build Redis payload (strip dataclass objects, keep dicts only)
            cache_data = {
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "universe_summary": summary,
                "signals": {},
            }

            for symbol, sig in all_signals.items():
                cache_data["signals"][symbol] = {
                    "trend": sig["trend"],
                    "momentum": sig["momentum"],
                    "levels": sig["levels"],
                    "volume": sig.get("volume", {}),
                    "relative_strength": sig.get("relative_strength", {}),
                }

            await self.state.client.set(
                "ta:current",
                json.dumps(cache_data, default=str),
                ex=50400,  # 14 hours TTL, same as macro
            )
            logger.info(f"Cached TA results for {len(all_signals)} symbols in Redis")
        except Exception as e:
            logger.warning(f"Failed to cache TA results: {e}")

    async def _persist_signals(self, all_signals: dict):
        """Persist signals to PostgreSQL for prediction tracking."""
        if self.cache.db is None:
            return

        today = date.today()
        persisted = 0

        # Read current macro regime from Redis for cross-referencing
        current_macro_regime = None
        try:
            if self.state:
                regime_data = await self.state.get_regime()
                if regime_data:
                    current_macro_regime = regime_data.get("regime", {}).get("primary")
        except Exception:
            pass  # Non-blocking — persist without macro regime if unavailable

        try:
            for symbol, sig in all_signals.items():
                trend = sig["trend"]
                mom = sig["momentum"]
                levels = sig["levels"]

                # Build full JSON for JSONB column
                full_json = {
                    "trend": trend,
                    "momentum": mom,
                    "levels": levels,
                }

                await self.cache.db.execute(
                    """
                    INSERT INTO technical_signals (
                        symbol, signal_date,
                        trend_regime, trend_score, adx, ma_alignment, ichimoku_signal, trend_maturity,
                        momentum_score, rsi_14, rsi_zone, macd_direction, macd_cross, divergence_type,
                        nearest_support, nearest_resistance, risk_reward_ratio,
                        full_signals_json, macro_regime
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                    )
                    ON CONFLICT (symbol, signal_date) DO UPDATE SET
                        trend_regime = EXCLUDED.trend_regime,
                        trend_score = EXCLUDED.trend_score,
                        adx = EXCLUDED.adx,
                        ma_alignment = EXCLUDED.ma_alignment,
                        ichimoku_signal = EXCLUDED.ichimoku_signal,
                        trend_maturity = EXCLUDED.trend_maturity,
                        momentum_score = EXCLUDED.momentum_score,
                        rsi_14 = EXCLUDED.rsi_14,
                        rsi_zone = EXCLUDED.rsi_zone,
                        macd_direction = EXCLUDED.macd_direction,
                        macd_cross = EXCLUDED.macd_cross,
                        divergence_type = EXCLUDED.divergence_type,
                        nearest_support = EXCLUDED.nearest_support,
                        nearest_resistance = EXCLUDED.nearest_resistance,
                        risk_reward_ratio = EXCLUDED.risk_reward_ratio,
                        full_signals_json = EXCLUDED.full_signals_json,
                        macro_regime = EXCLUDED.macro_regime
                    """,
                    symbol, today,
                    trend.get("trend_regime"),
                    trend.get("trend_score"),
                    trend.get("adx", {}).get("value"),
                    trend.get("ma", {}).get("alignment"),
                    trend.get("ichimoku", {}).get("signal"),
                    trend.get("maturity", {}).get("stage"),
                    mom.get("momentum_score"),
                    mom.get("rsi", {}).get("value"),
                    mom.get("rsi", {}).get("zone"),
                    mom.get("macd", {}).get("direction"),
                    mom.get("macd", {}).get("cross"),
                    mom.get("divergence", {}).get("type"),
                    levels.get("nearest_support"),
                    levels.get("nearest_resistance"),
                    min(levels.get("risk_reward_ratio") or 0, 999.99),
                    json.dumps(full_json, default=str),
                    current_macro_regime,
                )
                persisted += 1

            logger.info(f"Persisted {persisted} TA signals to PostgreSQL")
        except Exception as e:
            logger.warning(f"Failed to persist TA signals: {e}")

    def _build_opinion(
        self, all_signals: dict, summary: dict, query: AgentQuery, skipped: int,
    ) -> AgentOpinion:
        """Build AgentOpinion from computed signals."""
        # Overall direction from summary
        bull_pct = summary["pct_bullish"]
        bear_pct = summary["pct_bearish"]

        if bull_pct >= 60:
            direction = SignalDirection.BULLISH
        elif bull_pct >= 45 and bear_pct < 25:
            direction = SignalDirection.SLIGHTLY_BULLISH
        elif bear_pct >= 60:
            direction = SignalDirection.BEARISH
        elif bear_pct >= 45 and bull_pct < 25:
            direction = SignalDirection.SLIGHTLY_BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        # Confidence from trend agreement
        total = summary["total_analyzed"]
        max_pct = max(bull_pct, bear_pct, summary["pct_neutral"])
        if max_pct >= 70:
            confidence = ConfidenceLevel.HIGH
        elif max_pct >= 55:
            confidence = ConfidenceLevel.MODERATE
        else:
            confidence = ConfidenceLevel.LOW

        # Key findings
        key_findings = [
            f"Analyzed {total} symbols ({skipped} skipped for insufficient data)",
            f"Trend distribution: {bull_pct:.0f}% bullish, {bear_pct:.0f}% bearish, {summary['pct_neutral']:.0f}% neutral",
        ]

        if summary["divergences"]:
            div_summary = ", ".join(
                f"{d['symbol']} ({d['type']})" for d in summary["divergences"][:5]
            )
            key_findings.append(f"Active divergences: {div_summary}")

        # Build summary text
        summary_text = (
            f"Technical analysis of {total} symbols shows "
            f"{bull_pct:.0f}% bullish, {bear_pct:.0f}% bearish, "
            f"{summary['pct_neutral']:.0f}% neutral. "
        )
        if summary["divergences"]:
            summary_text += (
                f"{len(summary['divergences'])} momentum divergences detected. "
            )

        return AgentOpinion(
            agent_name=self.name,
            agent_layer=self.layer,
            direction=direction,
            confidence=confidence,
            summary=summary_text,
            key_findings=key_findings,
            data_points={
                "universe_summary": summary,
                "signals": {
                    sym: sig for sym, sig in all_signals.items()
                },
            },
            methodology=(
                "Multi-timeframe technical analysis using pandas-ta. "
                f"Trend analysis: Ichimoku Cloud, ADX, MA constellation (8/21/50/200). "
                f"Momentum: RSI (regime-adaptive thresholds), MACD, Stochastic. "
                f"Levels: Classic/Fibonacci pivots, Bollinger Bands, Donchian channels. "
                f"Divergence detection across RSI and price action."
            ),
            risks_to_view=[
                "Technical signals can fail in regime transitions",
                "Stale Polygon data may affect signal freshness",
            ],
            watch_items=[
                d["symbol"] + f" ({d['type']} divergence)"
                for d in summary.get("divergences", [])[:5]
            ],
            data_freshness=datetime.now(timezone.utc),
        )

    def _build_data_notes(self, all_signals: dict, summary: dict) -> str:
        """
        Build data notes string for Director cached context.

        Every computed value that Claude might cite must be explicitly surfaced
        with boundary notes to prevent fabrication.
        """
        parts = []
        parts.append("TECHNICAL ANALYSIS (Python-computed — cite EXACT values only):")
        parts.append(f"  Analyzed {summary.get('total_analyzed', 0)} symbols")
        parts.append(f"  Bullish: {summary.get('pct_bullish', 0):.0f}% | "
                     f"Bearish: {summary.get('pct_bearish', 0):.0f}% | "
                     f"Neutral: {summary.get('pct_neutral', 0):.0f}%")

        # Top movers by trend score
        scored = []
        for sym, sig in all_signals.items():
            ts = sig.get("trend", {}).get("trend_score", 0)
            ms = sig.get("momentum", {}).get("momentum_score", 0)
            vs = sig.get("volume", {}).get("volume_confirmation_score", 0)
            rs_s = sig.get("relative_strength", {}).get("rs_score", 0)
            scored.append((sym, ts, ms, vs, rs_s))

        scored.sort(key=lambda x: x[1] + x[2], reverse=True)

        if scored:
            parts.append("")
            parts.append("  TOP BULLISH (by trend + momentum):")
            for sym, ts, ms, vs, rs_s in scored[:5]:
                sig = all_signals[sym]
                regime = sig.get("trend", {}).get("trend_regime", "?")
                rsi = sig.get("momentum", {}).get("rsi", {}).get("value", "?")
                div = sig.get("momentum", {}).get("divergence", {}).get("type", "none")
                vol_conf = sig.get("volume", {}).get("volume_confirmation_score", 0)
                rs_regime = sig.get("relative_strength", {}).get("rs_regime", "?")
                div_str = f" [{div}]" if div != "none" else ""
                parts.append(f"    {sym}: {regime} (trend={ts:.2f}, mom={ms:.2f}, "
                           f"RSI={rsi}, vol_conf={vol_conf:.2f}, RS={rs_regime}){div_str}")

            parts.append("  TOP BEARISH:")
            for sym, ts, ms, vs, rs_s in scored[-5:]:
                sig = all_signals[sym]
                regime = sig.get("trend", {}).get("trend_regime", "?")
                rsi = sig.get("momentum", {}).get("rsi", {}).get("value", "?")
                div = sig.get("momentum", {}).get("divergence", {}).get("type", "none")
                vol_conf = sig.get("volume", {}).get("volume_confirmation_score", 0)
                rs_regime = sig.get("relative_strength", {}).get("rs_regime", "?")
                div_str = f" [{div}]" if div != "none" else ""
                parts.append(f"    {sym}: {regime} (trend={ts:.2f}, mom={ms:.2f}, "
                           f"RSI={rsi}, vol_conf={vol_conf:.2f}, RS={rs_regime}){div_str}")

        # Active divergences
        divs = summary.get("divergences", [])
        if divs:
            parts.append("")
            parts.append("  ACTIVE DIVERGENCES:")
            for d in divs[:10]:
                parts.append(f"    {d['symbol']}: {d['type']}")

        parts.append("")
        parts.append("  TA DATA BOUNDARY: ONLY cite TA signals listed above.")
        parts.append("  Do NOT invent trend regimes, RSI values, support/resistance levels,")
        parts.append("  volume scores, or relative strength metrics for symbols not listed.")

        return "\n".join(parts)

    def _empty_opinion(self, reason: str) -> AgentOpinion:
        """Return an empty opinion when computation can't proceed."""
        return AgentOpinion(
            agent_name=self.name,
            agent_layer=self.layer,
            direction=SignalDirection.NEUTRAL,
            confidence=ConfidenceLevel.VERY_LOW,
            summary=reason,
            key_findings=[reason],
            data_points={},
            methodology="No computation performed",
            risks_to_view=[reason],
            watch_items=[],
            data_freshness=datetime.now(timezone.utc),
        )

    async def health_check(self) -> HealthStatus:
        """Check if the TA agent is operational."""
        try:
            # Check if we have any price data
            count = await self.cache.db.fetchval(
                "SELECT COUNT(DISTINCT symbol) FROM market_data_daily"
            )
            return HealthStatus(
                agent_name=self.name,
                is_healthy=count > 0,
                last_computation=self._last_computation,
                data_freshness=self._last_computation,
                details={"symbols_available": count},
            )
        except Exception as e:
            return HealthStatus(
                agent_name=self.name,
                is_healthy=False,
                error_message=str(e),
            )
