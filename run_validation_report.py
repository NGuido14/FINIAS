"""
FINIAS TA Validation Report — Standalone with extended timeouts.
Run: python run_validation_report.py
"""
import asyncio
import asyncpg


def fmt(val):
    if val is None:
        return "     N/A"
    return "{:>7.2f}%".format(val)


async def main():
    conn = await asyncpg.connect(
        "postgresql://finias_agency:finias@localhost:5434/finias_agency",
        timeout=600,
        command_timeout=600,
    )

    print("=" * 70)
    print("  FINIAS TA SIGNAL VALIDATION REPORT - ALL 7 MODULES")
    print("=" * 70)

    total = await conn.fetchval("SELECT COUNT(*) FROM technical_signals")
    with_fwd = await conn.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE fwd_return_20d IS NOT NULL"
    )
    with_synth = await conn.fetchval(
        "SELECT COUNT(*) FROM technical_signals WHERE full_signals_json ? 'synthesis'"
    )
    print("")
    print("  Total signals: {:,}".format(total))
    print("  With forward returns: {:,}".format(with_fwd))
    print("  With synthesis data: {:,}".format(with_synth))

    # Section 1
    print("")
    print("-" * 70)
    print("  1. TREND REGIME -> RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "SELECT trend_regime, COUNT(*) as n, "
        "ROUND((AVG(fwd_return_5d)*100)::numeric,2) as avg_5d, "
        "ROUND((AVG(fwd_return_20d)*100)::numeric,2) as avg_20d, "
        "ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fwd_return_20d)*100)::numeric,2) as med_20d, "
        "ROUND((AVG(fwd_return_60d)*100)::numeric,2) as avg_60d, "
        "ROUND((AVG(fwd_max_drawdown_20d)*100)::numeric,2) as avg_dd "
        "FROM technical_signals WHERE fwd_return_20d IS NOT NULL "
        "GROUP BY trend_regime ORDER BY avg_20d DESC"
    )
    print("")
    print("  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Regime", "N", "5d", "20d", "Med20d", "60d", "DD"
    ))
    for r in rows:
        regime = r["trend_regime"] or "?"
        print("  {:<22} {:>8,} {} {} {} {} {}".format(
            regime, r["n"], fmt(r["avg_5d"]), fmt(r["avg_20d"]),
            fmt(r["med_20d"]), fmt(r["avg_60d"]), fmt(r["avg_dd"])
        ))

    # Section 2
    print("")
    print("-" * 70)
    print("  2. SYNTHESIS SETUP -> RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "SELECT "
        "full_signals_json->'synthesis'->'setup'->>'type' as setup, "
        "COUNT(*) as n, "
        "ROUND((AVG(fwd_return_5d)*100)::numeric,2) as avg_5d, "
        "ROUND((AVG(fwd_return_20d)*100)::numeric,2) as avg_20d, "
        "ROUND((AVG(fwd_return_60d)*100)::numeric,2) as avg_60d "
        "FROM technical_signals "
        "WHERE fwd_return_20d IS NOT NULL "
        "AND full_signals_json ? 'synthesis' "
        "GROUP BY setup ORDER BY avg_20d DESC"
    )
    print("")
    print("  {:<26} {:>8} {:>8} {:>8} {:>8}".format("Setup", "N", "5d", "20d", "60d"))
    for r in rows:
        setup = r["setup"] or "none"
        print("  {:<26} {:>8,} {} {} {}".format(
            setup, r["n"], fmt(r["avg_5d"]), fmt(r["avg_20d"]), fmt(r["avg_60d"])
        ))

    # Section 3
    print("")
    print("-" * 70)
    print("  3. SYNTHESIS SETUP x MACRO -> EXCESS RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "WITH spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM technical_signals WHERE symbol = $1"
        ") "
        "SELECT t.macro_regime, "
        "  full_signals_json->'synthesis'->'setup'->>'type' as setup, "
        "  COUNT(*) as n, "
        "  ROUND((AVG(t.fwd_return_20d - s.spy_20d)*100)::numeric,2) as excess "
        "FROM technical_signals t "
        "JOIN spy s ON t.signal_date = s.signal_date "
        "WHERE t.fwd_return_20d IS NOT NULL AND t.symbol != $1 "
        "  AND full_signals_json ? 'synthesis' "
        "  AND full_signals_json->'synthesis'->'setup'->>'type' != 'none' "
        "  AND t.macro_regime IS NOT NULL "
        "GROUP BY t.macro_regime, setup "
        "HAVING COUNT(*) >= 30 "
        "ORDER BY t.macro_regime, excess DESC",
        "SPY",
    )
    print("")
    print("  {:<16} {:<26} {:>7} {:>8}".format("Macro", "Setup", "N", "Excess"))
    cur_macro = None
    for r in rows:
        if r["macro_regime"] != cur_macro:
            if cur_macro:
                print("")
            cur_macro = r["macro_regime"]
        setup = r["setup"] or "none"
        print("  {:<16} {:<26} {:>7,} {}".format(
            r["macro_regime"], setup, r["n"], fmt(r["excess"])
        ))

    # Section 4
    print("")
    print("-" * 70)
    print("  4. CONVICTION LEVEL -> RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "WITH spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM technical_signals WHERE symbol = $1"
        ") "
        "SELECT "
        "  full_signals_json->'synthesis'->'conviction'->>'level' as conv, "
        "  COUNT(*) as n, "
        "  ROUND((AVG(t.fwd_return_20d)*100)::numeric,2) as avg_20d, "
        "  ROUND((AVG(t.fwd_return_20d - s.spy_20d)*100)::numeric,2) as excess, "
        "  ROUND((SUM(CASE WHEN t.fwd_return_20d - s.spy_20d > 0 THEN 1 ELSE 0 END)::numeric "
        "         / COUNT(*) * 100)::numeric,1) as win_rate "
        "FROM technical_signals t "
        "JOIN spy s ON t.signal_date = s.signal_date "
        "WHERE t.fwd_return_20d IS NOT NULL AND t.symbol != $1 "
        "  AND full_signals_json ? 'synthesis' "
        "GROUP BY conv ORDER BY excess DESC",
        "SPY",
    )
    print("")
    print("  {:<14} {:>8} {:>8} {:>8} {:>8}".format("Conviction", "N", "20d", "Excess", "WinRate"))
    for r in rows:
        conv = r["conv"] or "?"
        print("  {:<14} {:>8,} {} {} {}".format(
            conv, r["n"], fmt(r["avg_20d"]), fmt(r["excess"]), fmt(r["win_rate"])
        ))

    # Section 5
    print("")
    print("-" * 70)
    print("  5. VOLUME CONFIRMATION -> RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "WITH spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM technical_signals WHERE symbol = $1"
        "), binned AS ("
        "  SELECT t.*, s.spy_20d, "
        "    CASE "
        "      WHEN (full_signals_json->'volume'->>'volume_confirmation_score')::float > 0.2 THEN 'confirming' "
        "      WHEN (full_signals_json->'volume'->>'volume_confirmation_score')::float < -0.2 THEN 'contradicting' "
        "      ELSE 'neutral' "
        "    END as vol_bucket "
        "  FROM technical_signals t "
        "  JOIN spy s ON t.signal_date = s.signal_date "
        "  WHERE t.fwd_return_20d IS NOT NULL AND t.symbol != $1 "
        "    AND full_signals_json ? 'volume' "
        "    AND full_signals_json->'volume'->>'volume_confirmation_score' IS NOT NULL"
        ") "
        "SELECT vol_bucket, COUNT(*) as n, "
        "  ROUND((AVG(fwd_return_20d - spy_20d)*100)::numeric,2) as excess, "
        "  ROUND((SUM(CASE WHEN fwd_return_20d - spy_20d > 0 THEN 1 ELSE 0 END)::numeric "
        "         / COUNT(*) * 100)::numeric,1) as win_rate "
        "FROM binned GROUP BY vol_bucket ORDER BY excess DESC",
        "SPY",
    )
    print("")
    print("  {:<16} {:>8} {:>8} {:>8}".format("Volume", "N", "Excess", "WinRate"))
    for r in rows:
        print("  {:<16} {:>8,} {} {}".format(
            r["vol_bucket"], r["n"], fmt(r["excess"]), fmt(r["win_rate"])
        ))

    # Section 6
    print("")
    print("-" * 70)
    print("  6. RELATIVE STRENGTH REGIME -> RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "WITH spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM technical_signals WHERE symbol = $1"
        ") "
        "SELECT "
        "  full_signals_json->'relative_strength'->>'rs_regime' as rs, "
        "  COUNT(*) as n, "
        "  ROUND((AVG(t.fwd_return_20d - s.spy_20d)*100)::numeric,2) as excess, "
        "  ROUND((SUM(CASE WHEN t.fwd_return_20d - s.spy_20d > 0 THEN 1 ELSE 0 END)::numeric "
        "         / COUNT(*) * 100)::numeric,1) as win_rate "
        "FROM technical_signals t "
        "JOIN spy s ON t.signal_date = s.signal_date "
        "WHERE t.fwd_return_20d IS NOT NULL AND t.symbol != $1 "
        "  AND full_signals_json ? 'relative_strength' "
        "GROUP BY rs ORDER BY excess DESC",
        "SPY",
    )
    print("")
    print("  {:<16} {:>8} {:>8} {:>8}".format("RS Regime", "N", "Excess", "WinRate"))
    for r in rows:
        rs = r["rs"] or "?"
        print("  {:<16} {:>8,} {} {}".format(rs, r["n"], fmt(r["excess"]), fmt(r["win_rate"])))

    # Section 7
    print("")
    print("-" * 70)
    print("  7. SQUEEZE STATUS -> RETURNS")
    print("-" * 70)
    rows = await conn.fetch(
        "WITH spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM technical_signals WHERE symbol = $1"
        ") "
        "SELECT "
        "  CASE "
        "    WHEN (full_signals_json->'volatility'->'squeeze'->>'just_released')::boolean THEN 'just_released' "
        "    WHEN (full_signals_json->'volatility'->'squeeze'->>'active')::boolean THEN 'active' "
        "    ELSE 'none' "
        "  END as sq, "
        "  COUNT(*) as n, "
        "  ROUND((AVG(ABS(t.fwd_return_20d))*100)::numeric,2) as avg_abs, "
        "  ROUND((AVG(t.fwd_return_20d - s.spy_20d)*100)::numeric,2) as excess "
        "FROM technical_signals t "
        "JOIN spy s ON t.signal_date = s.signal_date "
        "WHERE t.fwd_return_20d IS NOT NULL AND t.symbol != $1 "
        "  AND full_signals_json ? 'volatility' "
        "GROUP BY sq ORDER BY avg_abs DESC",
        "SPY",
    )
    print("")
    print("  {:<16} {:>8} {:>8} {:>8}".format("Squeeze", "N", "|20d|", "Excess"))
    for r in rows:
        print("  {:<16} {:>8,} {} {}".format(r["sq"], r["n"], fmt(r["avg_abs"]), fmt(r["excess"])))

    # Section 8
    print("")
    print("-" * 70)
    print("  8. SYNTHESIS ACTION -> RETURNS (THE ULTIMATE TEST)")
    print("-" * 70)
    rows = await conn.fetch(
        "WITH spy AS ("
        "  SELECT signal_date, fwd_return_20d as spy_20d "
        "  FROM technical_signals WHERE symbol = $1"
        ") "
        "SELECT "
        "  full_signals_json->'synthesis'->>'action' as action, "
        "  COUNT(*) as n, "
        "  ROUND((AVG(t.fwd_return_20d)*100)::numeric,2) as avg_20d, "
        "  ROUND((AVG(t.fwd_return_20d - s.spy_20d)*100)::numeric,2) as excess, "
        "  ROUND((SUM(CASE WHEN t.fwd_return_20d > 0 THEN 1 ELSE 0 END)::numeric "
        "         / COUNT(*) * 100)::numeric,1) as win_rate, "
        "  ROUND((AVG(fwd_max_drawdown_20d)*100)::numeric,2) as avg_dd "
        "FROM technical_signals t "
        "JOIN spy s ON t.signal_date = s.signal_date "
        "WHERE t.fwd_return_20d IS NOT NULL AND t.symbol != $1 "
        "  AND full_signals_json ? 'synthesis' "
        "GROUP BY action ORDER BY excess DESC",
        "SPY",
    )
    print("")
    print("  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Action", "N", "20d", "Excess", "WinRate", "DD"
    ))
    for r in rows:
        action = r["action"] or "?"
        print("  {:<14} {:>8,} {} {} {} {}".format(
            action, r["n"], fmt(r["avg_20d"]), fmt(r["excess"]),
            fmt(r["win_rate"]), fmt(r["avg_dd"])
        ))

    print("")
    print("=" * 70)
    print("  REPORT COMPLETE")
    print("=" * 70)

    await conn.close()


asyncio.run(main())