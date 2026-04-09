"""
FINIAS TA Signal Backfill — Full 7-Module Daily

Walks through historical dates at daily intervals, computes ALL 7 TA
modules for all symbols AS OF each date (no look-ahead), stores in
technical_signals with full JSONB, then computes forward returns.

Usage:
    python -m finias.scripts.backfill_ta_signals                     # Full daily backfill
    python -m finias.scripts.backfill_ta_signals --weeks 52          # Last year only
    python -m finias.scripts.backfill_ta_signals --step 5            # Every 5 days (faster test)
    python -m finias.scripts.backfill_ta_signals --symbols SPY AAPL  # Specific symbols
    python -m finias.scripts.backfill_ta_signals --report            # Validation report only

Performance: ~4-8 hours for full daily backfill (1000 days × 500 symbols × 7 modules).
Cost: $0.00 (pure Python, no API calls).
"""

from __future__ import annotations
import asyncio
import logging
import sys
import json
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations

logger = logging.getLogger("finias.scripts.backfill_ta")

# Computation imports
from finias.agents.technical_analyst.computations.trend import analyze_trend
from finias.agents.technical_analyst.computations.momentum import analyze_momentum
from finias.agents.technical_analyst.computations.levels import analyze_levels
from finias.agents.technical_analyst.computations.volume import analyze_volume
from finias.agents.technical_analyst.computations.relative_strength import (
    analyze_relative_strength,
    compute_universe_returns,
    SECTOR_ETF_MAP,
)
from finias.agents.technical_analyst.computations.ta_volatility import (
    analyze_volatility as analyze_ta_volatility,
)
from finias.agents.technical_analyst.computations.signals import synthesize_signals

DEFAULT_WEEKS = 208  # ~4 years (full history)
STEP_DAYS = 1  # Daily
MIN_BARS = 200  # Need 200 bars for 200-day MA
LOOKBACK_BARS = 504  # ~2 years of trading days per computation


# ====================================================================
# MAIN BACKFILL
# ====================================================================

async def backfill_signals(
    db: DatabasePool,
    weeks: int = DEFAULT_WEEKS,
    step_days: int = STEP_DAYS,
    symbols: list[str] = None,
    skip_existing: bool = False,
) -> dict:
    """
    Walk through historical dates and compute ALL 7 TA modules.

    Note: skip_existing is False by default because we want to UPGRADE
    existing 3-module weekly signals with full 7-module data.
    """
    # Get symbol list
    if symbols is None:
        rows = await db.fetch(
            "SELECT DISTINCT symbol FROM symbol_universe WHERE is_active ORDER BY symbol"
        )
        symbols = [r["symbol"] for r in rows]
        if not symbols:
            rows = await db.fetch("SELECT DISTINCT symbol FROM market_data_daily ORDER BY symbol")
            symbols = [r["symbol"] for r in rows]

    print(f"  → Backfilling {weeks} weeks (step={step_days}d) for {len(symbols)} symbols...")

    # Determine date range
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(weeks=weeks)

    # Load ALL price data once
    print(f"  → Loading price data...")
    data_start = start_date - timedelta(days=LOOKBACK_BARS * 2)
    all_bars = await _load_all_bars(db, symbols, data_start, end_date)
    total_bars = sum(len(v) for v in all_bars.values())
    print(f"    Loaded {total_bars:,} bars for {len(all_bars)} symbols")

    # Load sector mapping
    sector_map = await _load_sector_map(db)
    print(f"    Loaded sector mapping for {len(sector_map)} symbols")

    # Get trading dates from SPY (ensures we only compute on actual trading days)
    trading_dates = _get_trading_dates(all_bars, start_date, end_date, step_days)
    print(f"    {len(trading_dates)} computation dates ({start_date} → {end_date})")

    # Get existing signal dates to potentially skip
    existing = set()
    if skip_existing:
        rows = await db.fetch("SELECT DISTINCT signal_date FROM technical_signals")
        existing = {r["signal_date"] for r in rows}
        if existing:
            print(f"    Skipping {len(existing)} dates with existing signals")

    # Walk through dates
    total_dates = 0
    total_signals = 0
    skipped = 0
    start_time = time.time()

    for i, current_date in enumerate(trading_dates):
        if current_date in existing:
            skipped += 1
            continue

        # Compute signals for this date (all 7 modules)
        date_signals = _compute_all_modules(
            all_bars, symbols, current_date, sector_map,
        )

        if date_signals:
            await _persist_batch(db, date_signals, current_date)
            total_signals += len(date_signals)

        total_dates += 1

        # Progress reporting
        if total_dates % 50 == 0:
            elapsed = time.time() - start_time
            rate = total_dates / elapsed if elapsed > 0 else 0
            remaining = (len(trading_dates) - i) / rate / 60 if rate > 0 else 0
            print(f"    [{total_dates}/{len(trading_dates)} dates] "
                  f"{total_signals:,} signals | "
                  f"{rate:.1f} dates/sec | "
                  f"~{remaining:.0f} min remaining")

    elapsed = time.time() - start_time
    print(f"  → Backfill complete: {total_dates} dates, {total_signals:,} signals "
          f"in {elapsed/60:.1f} minutes")

    return {
        "dates_computed": total_dates,
        "dates_skipped": skipped,
        "signals_stored": total_signals,
        "symbols": len(symbols),
    }


def _get_trading_dates(
    all_bars: dict, start_date: date, end_date: date, step_days: int,
) -> list[date]:
    """Get actual trading dates from SPY bars (no weekends/holidays)."""
    spy_df = all_bars.get("SPY")
    if spy_df is None:
        # Fallback: use any available symbol
        spy_df = next(iter(all_bars.values()))

    all_dates = sorted(spy_df["trade_date"].unique())
    filtered = [d for d in all_dates
                if isinstance(d, date) and start_date <= d <= end_date]

    # Apply step (every Nth trading day)
    if step_days > 1:
        filtered = filtered[::step_days]

    return filtered


def _compute_all_modules(
    all_bars: dict[str, pd.DataFrame],
    symbols: list[str],
    as_of_date: date,
    sector_map: dict[str, str],
) -> list[dict]:
    """
    Compute ALL 7 TA modules for all symbols as of a given date.
    No look-ahead bias — only data on or before as_of_date.
    """
    results = []

    # Step 1: Prepare DataFrames sliced to as_of_date
    dfs = {}
    for symbol in symbols:
        if symbol not in all_bars:
            continue
        full_df = all_bars[symbol]
        df = full_df[full_df["trade_date"] <= as_of_date].copy()
        if len(df) < MIN_BARS:
            continue
        dfs[symbol] = df.tail(LOOKBACK_BARS).reset_index(drop=True)

    if not dfs:
        return results

    # Step 2: Compute universe 20d returns (needed for RS percentile)
    universe_returns = compute_universe_returns(dfs)

    # Step 3: Compute macro regime proxy (SPY trailing 60d)
    macro_regime = _compute_macro_proxy(dfs.get("SPY"))

    # Step 4: Prepare sector ETF DataFrames
    sector_etf_dfs = {}
    for etf in SECTOR_ETF_MAP.values():
        if etf in dfs:
            sector_etf_dfs[etf] = dfs[etf]
    spy_df = dfs.get("SPY")

    # Step 5: Run all 7 modules for each symbol
    for symbol, df in dfs.items():
        try:
            # Module 1: Trend
            trend = analyze_trend(df, symbol=symbol)

            # Module 2: Momentum (uses trend regime)
            momentum = analyze_momentum(
                df, symbol=symbol, trend_regime=trend.trend_regime,
            )

            # Module 3: Levels
            levels = analyze_levels(df, symbol=symbol)

            # Module 4: Volume (uses trend regime)
            vol = analyze_volume(
                df, symbol=symbol, trend_regime=trend.trend_regime,
            )

            # Module 5: Relative Strength (needs sector context)
            sector_name = sector_map.get(symbol, "unknown")
            sector_etf = SECTOR_ETF_MAP.get(sector_name)
            sector_etf_df = sector_etf_dfs.get(sector_etf) if sector_etf else None

            rs = analyze_relative_strength(
                df, symbol=symbol, sector=sector_name,
                sector_etf_df=sector_etf_df, spy_df=spy_df,
                universe_returns_20d=universe_returns,
            )

            # Module 6: Volatility
            ta_vol = analyze_ta_volatility(df, symbol=symbol)

            # Module 7: Synthesis (needs macro regime)
            synthesis = synthesize_signals(
                trend=trend.to_dict(),
                momentum=momentum.to_dict(),
                levels=levels.to_dict(),
                volume=vol.to_dict(),
                relative_strength=rs.to_dict(),
                volatility=ta_vol.to_dict(),
                symbol=symbol,
                macro_regime=macro_regime,
            )

            close_price = float(df["close"].iloc[-1])

            results.append({
                "symbol": symbol,
                "close_price": close_price,
                "trend": trend.to_dict(),
                "momentum": momentum.to_dict(),
                "levels": levels.to_dict(),
                "volume": vol.to_dict(),
                "relative_strength": rs.to_dict(),
                "volatility": ta_vol.to_dict(),
                "synthesis": synthesis.to_dict(),
            })

        except Exception as e:
            logger.debug(f"Computation failed for {symbol} on {as_of_date}: {e}")

    return results


def _compute_macro_proxy(spy_df: pd.DataFrame) -> str:
    """Compute macro regime proxy from SPY trailing 60d return."""
    if spy_df is None or len(spy_df) < 61:
        return "unknown"

    current = float(spy_df["close"].iloc[-1])
    past = float(spy_df["close"].iloc[-61])
    ret = current / past - 1

    if ret < -0.10:
        return "crisis"
    elif ret < -0.03:
        return "risk_off"
    elif ret > 0.10:
        return "risk_on"
    elif ret > 0.03:
        return "moderate_bull"
    else:
        return "transition"


# ====================================================================
# DATA LOADING
# ====================================================================

async def _load_all_bars(
    db: DatabasePool, symbols: list[str], from_date: date, to_date: date,
) -> dict[str, pd.DataFrame]:
    """Load all OHLCV bars into memory as DataFrames."""
    rows = await db.fetch(
        """
        SELECT symbol, trade_date, open, high, low, close, volume
        FROM market_data_daily
        WHERE symbol = ANY($1) AND trade_date BETWEEN $2 AND $3
        ORDER BY symbol, trade_date ASC
        """,
        symbols, from_date, to_date,
    )

    bars_by_symbol: dict[str, list] = {}
    for row in rows:
        sym = row["symbol"]
        if sym not in bars_by_symbol:
            bars_by_symbol[sym] = []
        bars_by_symbol[sym].append(dict(row))

    result = {}
    for sym, bars in bars_by_symbol.items():
        df = pd.DataFrame(bars)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        result[sym] = df

    return result


async def _load_sector_map(db: DatabasePool) -> dict[str, str]:
    """Load GICS sector mapping from symbol_universe."""
    try:
        rows = await db.fetch(
            "SELECT symbol, sector FROM symbol_universe WHERE is_active AND sector IS NOT NULL"
        )
        return {r["symbol"]: r["sector"] for r in rows}
    except Exception:
        return {}


# ====================================================================
# PERSISTENCE
# ====================================================================

def _fmt(val, decimals=2):
    """Safe formatter for values that might be None."""
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


async def _persist_batch(db: DatabasePool, signals: list[dict], signal_date: date):
    """Persist a batch of signals with full 7-module JSONB."""
    for sig in signals:
        trend = sig["trend"]
        mom = sig["momentum"]
        levels = sig["levels"]

        # Full 7-module JSON for JSONB column
        full_json = json.dumps({
            "trend": trend,
            "momentum": mom,
            "levels": levels,
            "volume": sig.get("volume", {}),
            "relative_strength": sig.get("relative_strength", {}),
            "volatility": sig.get("volatility", {}),
            "synthesis": sig.get("synthesis", {}),
        }, default=str)

        # Compute macro regime from synthesis output
        macro_regime = sig.get("synthesis", {}).get("macro", {}).get("regime", None)

        await db.execute(
            """
            INSERT INTO technical_signals (
                symbol, signal_date, close_price,
                trend_regime, trend_score, adx, ma_alignment, ichimoku_signal, trend_maturity,
                momentum_score, rsi_14, rsi_zone, macd_direction, macd_cross, divergence_type,
                nearest_support, nearest_resistance, risk_reward_ratio,
                full_signals_json, macro_regime
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
            )
            ON CONFLICT (symbol, signal_date) DO UPDATE SET
                close_price = EXCLUDED.close_price,
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
            sig["symbol"], signal_date, sig["close_price"],
            trend.get("trend_regime"),
            _fmt(trend.get("trend_score"), 4),
            _fmt(trend.get("adx", {}).get("value")),
            trend.get("ma", {}).get("alignment"),
            trend.get("ichimoku", {}).get("signal"),
            trend.get("maturity", {}).get("stage"),
            _fmt(mom.get("momentum_score"), 4),
            _fmt(mom.get("rsi", {}).get("value")),
            mom.get("rsi", {}).get("zone"),
            mom.get("macd", {}).get("direction"),
            mom.get("macd", {}).get("cross"),
            mom.get("divergence", {}).get("type"),
            _fmt(levels.get("nearest_support"), 4),
            _fmt(levels.get("nearest_resistance"), 4),
            min(_fmt(levels.get("risk_reward_ratio")) or 0, 999.99),
            full_json,
            macro_regime,
        )


# ====================================================================
# FORWARD RETURNS
# ====================================================================

async def compute_forward_returns(db: DatabasePool) -> dict:
    """
    Fill in forward returns for all signals from actual future prices.
    Handles 1d, 5d, 20d, 60d returns and 20d max drawdown.
    """
    updated = 0

    # Get symbols with missing forward returns, process one at a time to avoid timeout
    symbol_rows = await db.fetch(
        """
        SELECT DISTINCT symbol FROM technical_signals
        WHERE close_price IS NOT NULL
          AND (fwd_return_1d IS NULL OR fwd_return_5d IS NULL
               OR fwd_return_20d IS NULL OR fwd_return_60d IS NULL)
        """
    )

    if not symbol_rows:
        print("  → All forward returns already computed")
        return {"updated": 0}

    symbols_to_process = [r["symbol"] for r in symbol_rows]
    print(f"  → Computing forward returns for {len(symbols_to_process)} symbols...")

    # Fetch per-symbol to avoid timeout on large queries
    by_symbol: dict[str, list] = {}
    for sym in symbols_to_process:
        batch = await db.fetch(
            """
            SELECT id, symbol, signal_date, close_price
            FROM technical_signals
            WHERE symbol = $1
              AND close_price IS NOT NULL
              AND (fwd_return_1d IS NULL OR fwd_return_5d IS NULL
                   OR fwd_return_20d IS NULL OR fwd_return_60d IS NULL)
            ORDER BY signal_date ASC
            """,
            sym,
        )
        if batch:
            by_symbol[sym] = [dict(r) for r in batch]

    for sym, sigs in by_symbol.items():
        # Get all prices for this symbol
        prices = await db.fetch(
            """
            SELECT trade_date, close FROM market_data_daily
            WHERE symbol = $1 ORDER BY trade_date ASC
            """,
            sym,
        )
        if not prices:
            continue

        price_map = {r["trade_date"]: float(r["close"]) for r in prices}
        dates_sorted = sorted(price_map.keys())

        for sig in sigs:
            entry_price = float(sig["close_price"])
            sig_date = sig["signal_date"]

            # Find index of signal date
            try:
                idx = dates_sorted.index(sig_date)
            except ValueError:
                continue

            fwd_1d = fwd_5d = fwd_20d = fwd_60d = dd_20d = None

            # 1-day return
            if idx + 1 < len(dates_sorted):
                fwd_1d = price_map[dates_sorted[idx + 1]] / entry_price - 1

            # 5-day return
            if idx + 5 < len(dates_sorted):
                fwd_5d = price_map[dates_sorted[idx + 5]] / entry_price - 1

            # 20-day return + max drawdown
            if idx + 20 < len(dates_sorted):
                fwd_20d = price_map[dates_sorted[idx + 20]] / entry_price - 1
                # Max drawdown over 20-day window
                window_prices = [price_map[dates_sorted[idx + j]] for j in range(1, 21)]
                min_price = min(window_prices)
                dd_20d = min_price / entry_price - 1

            # 60-day return
            if idx + 60 < len(dates_sorted):
                fwd_60d = price_map[dates_sorted[idx + 60]] / entry_price - 1

            # Update
            await db.execute(
                """
                UPDATE technical_signals
                SET fwd_return_1d = COALESCE($2, fwd_return_1d),
                    fwd_return_5d = COALESCE($3, fwd_return_5d),
                    fwd_return_20d = COALESCE($4, fwd_return_20d),
                    fwd_return_60d = COALESCE($5, fwd_return_60d),
                    fwd_max_drawdown_20d = COALESCE($6, fwd_max_drawdown_20d)
                WHERE id = $1
                """,
                sig["id"],
                _fmt(fwd_1d, 4), _fmt(fwd_5d, 4), _fmt(fwd_20d, 4),
                _fmt(fwd_60d, 4), _fmt(dd_20d, 4),
            )
            updated += 1

    print(f"  → Forward returns computed for {updated:,} signals")
    return {"updated": updated}


# ====================================================================
# VALIDATION REPORT
# ====================================================================

async def generate_accuracy_report(db: DatabasePool):
    """
    Comprehensive validation report testing ALL modules against forward returns.
    """
    print("\n" + "=" * 70)
    print("  FINIAS TA SIGNAL VALIDATION REPORT — ALL 7 MODULES")
    print("=" * 70)

    total = await db.fetchval("SELECT COUNT(*) FROM technical_signals")
    with_fwd = await db.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE fwd_return_20d IS NOT NULL"
    )
    with_json = await db.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE full_signals_json != '{}'"
    )
    print(f"\n  Total signals: {total:,}")
    print(f"  With forward returns: {with_fwd:,}")
    print(f"  With full 7-module JSON: {with_json:,}")

    # ---- Section 1: Trend Regime → Returns ----
    print("\n" + "─" * 70)
    print("  1. FORWARD RETURNS BY TREND REGIME")
    print("─" * 70)

    rows = await db.fetch("""
        SELECT trend_regime,
            COUNT(*) as n,
            ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
            ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
            ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fwd_return_20d) * 100)::numeric, 2) as med_20d,
            ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d,
            ROUND((AVG(fwd_max_drawdown_20d) * 100)::numeric, 2) as avg_dd
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL
        GROUP BY trend_regime
        ORDER BY avg_20d DESC
    """)

    print(f"\n  {'Regime':<26} {'N':>7} {'Avg 5d':>8} {'Avg 20d':>8} {'Med 20d':>8} {'Avg 60d':>8} {'Avg DD':>8}")
    print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in rows:
        print(f"  {r['trend_regime'] or 'N/A':<26} {r['n']:>7,} "
              f"{_rfmt(r['avg_5d'])} {_rfmt(r['avg_20d'])} {_rfmt(r['med_20d'])} "
              f"{_rfmt(r['avg_60d'])} {_rfmt(r['avg_dd'])}")

    # ---- Section 2: Synthesis Setup → Returns ----
    print("\n" + "─" * 70)
    print("  2. FORWARD RETURNS BY SYNTHESIS SETUP TYPE")
    print("  (Does the synthesis engine identify profitable setups?)")
    print("─" * 70)

    rows = await db.fetch("""
        SELECT
            full_signals_json->'synthesis'->'setup'->>'type' as setup_type,
            COUNT(*) as n,
            ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
            ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
            ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d
        FROM technical_signals
        WHERE fwd_return_20d IS NOT NULL
          AND full_signals_json != '{}'
        GROUP BY setup_type
        ORDER BY avg_20d DESC
    """)

    print(f"\n  {'Setup':<26} {'N':>7} {'Avg 5d':>8} {'Avg 20d':>8} {'Avg 60d':>8}")
    print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for r in rows:
        print(f"  {r['setup_type'] or 'none':<26} {r['n']:>7,} "
              f"{_rfmt(r['avg_5d'])} {_rfmt(r['avg_20d'])} {_rfmt(r['avg_60d'])}")

    # ---- Section 3: Synthesis Setup + Macro Regime → Excess Returns ----
    print("\n" + "─" * 70)
    print("  3. SYNTHESIS SETUP × MACRO REGIME → EXCESS RETURNS")
    print("  (Does macro conditioning improve signal quality?)")
    print("─" * 70)

    rows = await db.fetch("""
        WITH spy AS (
            SELECT signal_date, fwd_return_20d as spy_20d
            FROM technical_signals WHERE symbol = $1
        )
        SELECT
            t.macro_regime,
            full_signals_json->'synthesis'->'setup'->>'type' as setup_type,
            COUNT(*) as n,
            ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess_20d
        FROM technical_signals t
        JOIN spy s ON t.signal_date = s.signal_date
        WHERE t.fwd_return_20d IS NOT NULL
          AND t.symbol != $1
          AND t.full_signals_json != '{}'
          AND t.macro_regime IS NOT NULL
          AND full_signals_json->'synthesis'->'setup'->>'type' != 'none'
        GROUP BY t.macro_regime, setup_type
        HAVING COUNT(*) >= 30
        ORDER BY t.macro_regime, excess_20d DESC
    """, "SPY")

    print(f"\n  {'Macro':<18} {'Setup':<26} {'N':>7} {'Excess 20d':>10}")
    print(f"  {'─'*18} {'─'*22} {'─'*7} {'─'*10}")
    current_macro = None
    for r in rows:
        if r["macro_regime"] != current_macro:
            if current_macro:
                print()
            current_macro = r["macro_regime"]
        print(f"  {r['macro_regime']:<18} {r['setup_type'] or 'none':<26} "
              f"{r['n']:>7,} {_rfmt(r['excess_20d'])}")

    # ---- Section 4: Conviction Score → Returns ----
    print("\n" + "─" * 70)
    print("  4. CONVICTION LEVEL → RETURNS")
    print("  (Do high conviction signals outperform?)")
    print("─" * 70)

    rows = await db.fetch("""
        WITH spy AS (
            SELECT signal_date, fwd_return_20d as spy_20d
            FROM technical_signals WHERE symbol = $1
        )
        SELECT
            full_signals_json->'synthesis'->'conviction'->>'level' as conv_level,
            COUNT(*) as n,
            ROUND((AVG(t.fwd_return_20d) * 100)::numeric, 2) as avg_20d,
            ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess_20d,
            ROUND((SUM(CASE WHEN t.fwd_return_20d - s.spy_20d > 0 THEN 1 ELSE 0 END)::numeric
                   / COUNT(*) * 100)::numeric, 1) as win_rate
        FROM technical_signals t
        JOIN spy s ON t.signal_date = s.signal_date
        WHERE t.fwd_return_20d IS NOT NULL
          AND t.symbol != $1
          AND t.full_signals_json != '{}'
        GROUP BY conv_level
        ORDER BY excess_20d DESC
    """, "SPY")

    print(f"\n  {'Conviction':<18} {'N':>7} {'Avg 20d':>8} {'Excess':>8} {'Win Rate':>9}")
    print(f"  {'─'*18} {'─'*7} {'─'*8} {'─'*8} {'─'*9}")
    for r in rows:
        print(f"  {r['conv_level'] or 'N/A':<18} {r['n']:>7,} "
              f"{_rfmt(r['avg_20d'])} {_rfmt(r['excess_20d'])} "
              f"{_rfmt(r['win_rate'], suffix='%')}")

    # ---- Section 5: Volume Confirmation → Returns ----
    print("\n" + "─" * 70)
    print("  5. VOLUME CONFIRMATION → RETURNS")
    print("  (Does volume confirmation improve signal quality?)")
    print("─" * 70)

    rows = await db.fetch("""
        WITH spy AS (
            SELECT signal_date, fwd_return_20d as spy_20d
            FROM technical_signals WHERE symbol = $1
        ),
        binned AS (
            SELECT t.*,
                s.spy_20d,
                CASE
                    WHEN (full_signals_json->'volume'->>'volume_confirmation_score')::float > 0.2 THEN 'confirming'
                    WHEN (full_signals_json->'volume'->>'volume_confirmation_score')::float < -0.2 THEN 'contradicting'
                    ELSE 'neutral'
                END as vol_bucket
            FROM technical_signals t
            JOIN spy s ON t.signal_date = s.signal_date
            WHERE t.fwd_return_20d IS NOT NULL
              AND t.symbol != $1
              AND t.full_signals_json != '{}'
              AND full_signals_json->'volume'->>'volume_confirmation_score' IS NOT NULL
        )
        SELECT vol_bucket,
            COUNT(*) as n,
            ROUND((AVG(fwd_return_20d - spy_20d) * 100)::numeric, 2) as excess_20d,
            ROUND((SUM(CASE WHEN fwd_return_20d - spy_20d > 0 THEN 1 ELSE 0 END)::numeric
                   / COUNT(*) * 100)::numeric, 1) as win_rate
        FROM binned
        GROUP BY vol_bucket
        ORDER BY excess_20d DESC
    """, "SPY")

    print(f"\n  {'Volume':<18} {'N':>7} {'Excess 20d':>10} {'Win Rate':>9}")
    print(f"  {'─'*18} {'─'*7} {'─'*10} {'─'*9}")
    for r in rows:
        print(f"  {r['vol_bucket']:<18} {r['n']:>7,} "
              f"{_rfmt(r['excess_20d'])} {_rfmt(r['win_rate'], suffix='%')}")

    # ---- Section 6: RS Regime → Returns ----
    print("\n" + "─" * 70)
    print("  6. RELATIVE STRENGTH REGIME → RETURNS")
    print("  (Does RS improvement predict outperformance?)")
    print("─" * 70)

    rows = await db.fetch("""
        WITH spy AS (
            SELECT signal_date, fwd_return_20d as spy_20d
            FROM technical_signals WHERE symbol = $1
        )
        SELECT
            full_signals_json->'relative_strength'->>'rs_regime' as rs_regime,
            COUNT(*) as n,
            ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess_20d,
            ROUND((SUM(CASE WHEN t.fwd_return_20d - s.spy_20d > 0 THEN 1 ELSE 0 END)::numeric
                   / COUNT(*) * 100)::numeric, 1) as win_rate
        FROM technical_signals t
        JOIN spy s ON t.signal_date = s.signal_date
        WHERE t.fwd_return_20d IS NOT NULL
          AND t.symbol != $1
          AND t.full_signals_json != '{}'
          AND full_signals_json->'relative_strength'->>'rs_regime' IS NOT NULL
        GROUP BY rs_regime
        ORDER BY excess_20d DESC
    """, "SPY")

    print(f"\n  {'RS Regime':<18} {'N':>7} {'Excess 20d':>10} {'Win Rate':>9}")
    print(f"  {'─'*18} {'─'*7} {'─'*10} {'─'*9}")
    for r in rows:
        print(f"  {r['rs_regime'] or 'N/A':<18} {r['n']:>7,} "
              f"{_rfmt(r['excess_20d'])} {_rfmt(r['win_rate'], suffix='%')}")

    # ---- Section 7: Squeeze → Returns ----
    print("\n" + "─" * 70)
    print("  7. SQUEEZE STATUS → RETURNS")
    print("  (Do squeezes predict large moves?)")
    print("─" * 70)

    rows = await db.fetch("""
        WITH spy AS (
            SELECT signal_date, fwd_return_20d as spy_20d
            FROM technical_signals WHERE symbol = $1
        )
        SELECT
            CASE
                WHEN (full_signals_json->'volatility'->'squeeze'->>'just_released')::boolean THEN 'just_released'
                WHEN (full_signals_json->'volatility'->'squeeze'->>'active')::boolean THEN 'active_squeeze'
                ELSE 'no_squeeze'
            END as squeeze_status,
            COUNT(*) as n,
            ROUND((AVG(ABS(t.fwd_return_20d)) * 100)::numeric, 2) as avg_abs_20d,
            ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess_20d
        FROM technical_signals t
        JOIN spy s ON t.signal_date = s.signal_date
        WHERE t.fwd_return_20d IS NOT NULL
          AND t.symbol != $1
          AND t.full_signals_json != '{}'
          AND full_signals_json->'volatility'->'squeeze'->>'active' IS NOT NULL
        GROUP BY squeeze_status
        ORDER BY avg_abs_20d DESC
    """, "SPY")

    print(f"\n  {'Squeeze':<18} {'N':>7} {'Avg |20d|':>10} {'Excess 20d':>10}")
    print(f"  {'─'*18} {'─'*7} {'─'*10} {'─'*10}")
    for r in rows:
        print(f"  {r['squeeze_status']:<18} {r['n']:>7,} "
              f"{_rfmt(r['avg_abs_20d'])} {_rfmt(r['excess_20d'])}")

    # ---- Section 8: Action → Returns (The Ultimate Test) ----
    print("\n" + "─" * 70)
    print("  8. SYNTHESIS ACTION → RETURNS (THE ULTIMATE TEST)")
    print("  (Do 'buy' signals make money? Do 'sell' signals lose?)")
    print("─" * 70)

    rows = await db.fetch("""
        WITH spy AS (
            SELECT signal_date, fwd_return_20d as spy_20d
            FROM technical_signals WHERE symbol = $1
        )
        SELECT
            full_signals_json->'synthesis'->>'action' as action,
            COUNT(*) as n,
            ROUND((AVG(t.fwd_return_20d) * 100)::numeric, 2) as avg_20d,
            ROUND((AVG(t.fwd_return_20d - s.spy_20d) * 100)::numeric, 2) as excess_20d,
            ROUND((SUM(CASE WHEN t.fwd_return_20d > 0 THEN 1 ELSE 0 END)::numeric
                   / COUNT(*) * 100)::numeric, 1) as win_rate,
            ROUND((AVG(fwd_max_drawdown_20d) * 100)::numeric, 2) as avg_dd
        FROM technical_signals t
        JOIN spy s ON t.signal_date = s.signal_date
        WHERE t.fwd_return_20d IS NOT NULL
          AND t.symbol != $1
          AND t.full_signals_json != '{}'
          AND full_signals_json->'synthesis'->>'action' IS NOT NULL
        GROUP BY action
        ORDER BY excess_20d DESC
    """, "SPY")

    print(f"\n  {'Action':<18} {'N':>7} {'Avg 20d':>8} {'Excess':>8} {'Win Rate':>9} {'Avg DD':>8}")
    print(f"  {'─'*18} {'─'*7} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")
    for r in rows:
        print(f"  {r['action'] or 'N/A':<18} {r['n']:>7,} "
              f"{_rfmt(r['avg_20d'])} {_rfmt(r['excess_20d'])} "
              f"{_rfmt(r['win_rate'], suffix='%')} {_rfmt(r['avg_dd'])}")

    print("\n" + "=" * 70)


def _rfmt(val, suffix="%"):
    """Format a report value."""
    if val is None:
        return f"{'N/A':>8}"
    return f"{val:>7.2f}{suffix}"


# ====================================================================
# MAIN
# ====================================================================

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = set(sys.argv[1:])
    report_only = "--report" in args

    # Parse --weeks N
    weeks = DEFAULT_WEEKS
    if "--weeks" in sys.argv:
        idx = sys.argv.index("--weeks")
        if idx + 1 < len(sys.argv):
            weeks = int(sys.argv[idx + 1])

    # Parse --step N
    step_days = STEP_DAYS
    if "--step" in sys.argv:
        idx = sys.argv.index("--step")
        if idx + 1 < len(sys.argv):
            step_days = int(sys.argv[idx + 1])

    # Parse --symbols SYM1 SYM2
    symbols = None
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        symbols = [s.upper() for s in sys.argv[idx + 1:] if not s.startswith("--")]

    print("╔══════════════════════════════════════════╗")
    print("║   FINIAS TA Signal Backfill — 7 Modules  ║")
    print("╚══════════════════════════════════════════╝\n")

    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    try:
        if report_only:
            await generate_accuracy_report(db)
        else:
            # Step 1: Backfill signals with all 7 modules
            result = await backfill_signals(
                db, weeks=weeks, step_days=step_days, symbols=symbols,
            )
            print(f"\n  Backfill: {result['signals_stored']:,} signals for "
                  f"{result['symbols']} symbols over {result['dates_computed']} dates")

            # Step 2: Compute forward returns
            fwd = await compute_forward_returns(db)
            print(f"  Forward returns: {fwd['updated']:,} signals updated")

            # Step 3: Generate validation report
            await generate_accuracy_report(db)

    finally:
        await db.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
