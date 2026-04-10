"""
Re-run synthesis engine on all historical signals with recalibrated weights.

The underlying indicators (trend, momentum, levels, volume, RS, volatility)
haven't changed — only the synthesis engine weights and aligned_combos changed.
This script reads the stored module outputs from full_signals_json, re-runs
synthesize_signals(), and updates the synthesis portion.

Run: python resynthesize.py
Then: python run_validation_report.py
"""
import asyncio
import json
import asyncpg

from finias.agents.technical_analyst.computations.signals import synthesize_signals


async def main():
    conn = await asyncpg.connect(
        "postgresql://finias_agency:finias@localhost:5434/finias_agency",
        timeout=600,
        command_timeout=600,
    )

    # Count signals with synthesis data
    total = await conn.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE full_signals_json ? 'synthesis'"
    )
    print("Signals to re-synthesize: {:,}".format(total))

    # Also count those without synthesis (old 3-module backfill)
    no_synth = await conn.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE NOT (full_signals_json ? 'synthesis')"
    )
    if no_synth > 0:
        print("Signals without synthesis data (old backfill): {:,} — skipping".format(no_synth))

    # Process in batches by date
    dates = await conn.fetch(
        "SELECT DISTINCT signal_date FROM technical_signals "
        "WHERE full_signals_json ? 'synthesis' "
        "ORDER BY signal_date"
    )
    print("Dates to process: {}".format(len(dates)))

    updated = 0
    for i, row in enumerate(dates):
        sd = row["signal_date"]

        signals = await conn.fetch(
            "SELECT id, symbol, macro_regime, full_signals_json "
            "FROM technical_signals "
            "WHERE signal_date = $1 AND full_signals_json ? 'synthesis'",
            sd,
        )

        for sig in signals:
            fj = json.loads(sig["full_signals_json"])

            trend = fj.get("trend", {})
            momentum = fj.get("momentum", {})
            levels = fj.get("levels", {})
            volume = fj.get("volume", {})
            rs = fj.get("relative_strength", {})
            vol = fj.get("volatility", {})
            macro = sig["macro_regime"] or "unknown"

            # Re-run synthesis with NEW weights
            synthesis = synthesize_signals(
                trend=trend,
                momentum=momentum,
                levels=levels,
                volume=volume,
                relative_strength=rs,
                volatility=vol,
                symbol=sig["symbol"],
                macro_regime=macro,
            )

            # Update the JSON with new synthesis
            fj["synthesis"] = synthesis.to_dict()

            await conn.execute(
                "UPDATE technical_signals SET full_signals_json = $1 WHERE id = $2",
                json.dumps(fj, default=str),
                sig["id"],
            )
            updated += 1

        if (i + 1) % 50 == 0:
            print("  [{}/{}] {:,} signals updated...".format(i + 1, len(dates), updated))

    print("\nTotal re-synthesized: {:,}".format(updated))

    # Quick check — action distribution
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