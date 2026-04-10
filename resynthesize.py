"""
Re-synthesize all historical signals with enhanced engine.

This version:
1. Loads price data for each symbol
2. Computes enhanced signals (ATR, RSI2, 52wk high, weekly trend, acceleration)
3. Re-runs synthesize_signals() with enhanced inputs
4. Updates full_signals_json in the database

Run: python resynthesize.py
Then: python run_validation_report.py
"""
import asyncio
import json
import time
import numpy as np
import pandas as pd
import asyncpg

from finias.agents.technical_analyst.computations.enhanced import compute_enhanced_signals
from finias.agents.technical_analyst.computations.signals import synthesize_signals


async def main():
    conn = await asyncpg.connect(
        "postgresql://finias_agency:finias@localhost:5434/finias_agency",
        timeout=600,
        command_timeout=600,
    )

    total = await conn.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE full_signals_json ? 'synthesis'"
    )
    print("Signals to re-synthesize: {:,}".format(total))

    # Load ALL price data into memory (keyed by symbol)
    print("Loading price data...")
    price_rows = await conn.fetch(
        "SELECT symbol, trade_date, open, high, low, close, volume "
        "FROM market_data_daily ORDER BY symbol, trade_date ASC"
    )
    print("  Loaded {:,} price bars".format(len(price_rows)))

    # Build DataFrames by symbol
    price_dfs = {}
    current_sym = None
    current_rows = []
    for row in price_rows:
        sym = row["symbol"]
        if sym != current_sym:
            if current_sym and current_rows:
                df = pd.DataFrame(current_rows)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                price_dfs[current_sym] = df
            current_sym = sym
            current_rows = []
        current_rows.append({
            "trade_date": row["trade_date"],
            "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"],
            "volume": row["volume"],
        })
    if current_sym and current_rows:
        df = pd.DataFrame(current_rows)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        price_dfs[current_sym] = df
    print("  Built DataFrames for {} symbols".format(len(price_dfs)))

    # Process by date
    dates = await conn.fetch(
        "SELECT DISTINCT signal_date FROM technical_signals "
        "WHERE full_signals_json ? 'synthesis' "
        "ORDER BY signal_date"
    )
    print("Dates to process: {}".format(len(dates)))

    updated = 0
    start_time = time.time()

    for i, row in enumerate(dates):
        sd = row["signal_date"]

        signals = await conn.fetch(
            "SELECT id, symbol, macro_regime, full_signals_json "
            "FROM technical_signals "
            "WHERE signal_date = $1 AND full_signals_json ? 'synthesis'",
            sd,
        )

        for sig in signals:
            symbol = sig["symbol"]
            fj = json.loads(sig["full_signals_json"])
            macro = sig["macro_regime"] or "unknown"

            trend = fj.get("trend", {})
            momentum = fj.get("momentum", {})
            levels = fj.get("levels", {})
            volume = fj.get("volume", {})
            rs = fj.get("relative_strength", {})
            vol = fj.get("volatility", {})

            # Compute enhanced signals from price data
            enhanced_dict = {}
            if symbol in price_dfs:
                full_df = price_dfs[symbol]
                # Slice to as-of date (no look-ahead)
                df_slice = full_df[full_df["trade_date"] <= sd].copy()
                if len(df_slice) >= 200:
                    df_slice = df_slice.tail(504).reset_index(drop=True)
                    daily_regime = trend.get("trend_regime", "unknown")
                    enhanced = compute_enhanced_signals(
                        df_slice, symbol=symbol, daily_trend_regime=daily_regime,
                    )
                    enhanced_dict = enhanced.to_dict()

            # Re-run synthesis with enhanced signals
            synthesis = synthesize_signals(
                trend=trend,
                momentum=momentum,
                levels=levels,
                volume=volume,
                relative_strength=rs,
                volatility=vol,
                symbol=symbol,
                macro_regime=macro,
                enhanced=enhanced_dict if enhanced_dict else None,
            )

            # Update JSON
            fj["enhanced"] = enhanced_dict
            fj["synthesis"] = synthesis.to_dict()

            await conn.execute(
                "UPDATE technical_signals SET full_signals_json = $1 WHERE id = $2",
                json.dumps(fj, default=str),
                sig["id"],
            )
            updated += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = updated / elapsed if elapsed > 0 else 0
            remaining = (len(dates) - i - 1) * (elapsed / (i + 1)) / 60
            print("  [{}/{}] {:,} signals | {:.0f}/sec | ~{:.0f} min remaining".format(
                i + 1, len(dates), updated, rate, remaining))

    elapsed = time.time() - start_time
    print("\nTotal re-synthesized: {:,} in {:.1f} minutes".format(updated, elapsed / 60))

    # Action distribution
    print("\nNew action distribution:")
    rows = await conn.fetch(
        "SELECT "
        "full_signals_json->'synthesis'->>'action' as action, "
        "COUNT(*) as n "
        "FROM technical_signals "
        "WHERE full_signals_json ? 'synthesis' "
        "GROUP BY action ORDER BY n DESC"
    )
    for r in rows:
        print("  {}: {:,}".format(r["action"], r["n"]))

    await conn.close()
    print("\nDone. Now run: python run_validation_report.py")


asyncio.run(main())