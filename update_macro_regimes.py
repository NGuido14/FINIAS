"""
Update technical_signals.macro_regime using real macro backtest results.
Processes one date at a time to avoid timeout.
"""
import asyncio
import asyncpg


async def main():
    conn = await asyncpg.connect(
        "postgresql://finias_agency:finias@localhost:5434/finias_agency",
        timeout=600,
        command_timeout=600,
    )

    regimes = await conn.fetch(
        "SELECT sim_date, primary_regime FROM backtest_results ORDER BY sim_date ASC"
    )
    print("Macro backtest results: {} dates".format(len(regimes)))
    print("  Range: {} to {}".format(regimes[0]["sim_date"], regimes[-1]["sim_date"]))

    regime_dates = [(r["sim_date"], r["primary_regime"]) for r in regimes]

    signal_dates = await conn.fetch(
        "SELECT DISTINCT signal_date FROM technical_signals ORDER BY signal_date"
    )
    print("Signal dates to update: {}".format(len(signal_dates)))

    updated = 0
    for i, row in enumerate(signal_dates):
        sd = row["signal_date"]
        best_regime = None
        for rd, regime in reversed(regime_dates):
            if rd <= sd:
                best_regime = regime
                break

        if best_regime is None:
            continue

        result = await conn.execute(
            "UPDATE technical_signals SET macro_regime = $1 WHERE signal_date = $2",
            best_regime, sd,
        )
        count = int(result.split(" ")[-1])
        updated += count

        if (i + 1) % 100 == 0:
            print("  [{}/{}] {} signals updated...".format(i + 1, len(signal_dates), updated))

    print("\nTotal updated: {:,}".format(updated))

    rows = await conn.fetch(
        "SELECT macro_regime, COUNT(*) as n FROM technical_signals "
        "WHERE macro_regime IS NOT NULL GROUP BY macro_regime ORDER BY n DESC"
    )
    print("\nRegime distribution:")
    for r in rows:
        print("  {}: {:,}".format(r["macro_regime"], r["n"]))

    null_count = await conn.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE macro_regime IS NULL"
    )
    print("\nStill NULL: {:,}".format(null_count))

    await conn.close()


asyncio.run(main())