"""
Technical Analyst History Query Tool.

Provides the Director with access to historical TA signal data from
the technical_signals PostgreSQL table. This is a lightweight tool
(not a registered agent) — just a DB query function wired into the
Director's chat() method.

Pattern matches finias/agents/macro_strategist/history.py.
"""

from __future__ import annotations
from datetime import date, timedelta
import logging

logger = logging.getLogger("finias.ta.history")


def get_ta_history_tool_definition() -> dict:
    """
    Claude tool_use definition for the TA history tool.
    """
    return {
        "name": "query_ta_history",
        "description": (
            "Query historical technical analysis signals from FINIAS databases. "
            "Use this when the user asks about past technical signals, historical "
            "trend regimes, signal accuracy, or how a stock's technical picture "
            "has evolved over time. This tool reads from pre-computed technical "
            "signals stored weekly for ~500 S&P 500 stocks over 2+ years. "
            "FREE and INSTANT — no API cost.\n\n"
            "Available queries:\n"
            "- symbol_history: How a specific symbol's signals evolved over time\n"
            "- divergence_scan: Find all divergences in a date range\n"
            "- regime_scan: Find symbols in a specific trend regime on a date\n"
            "- accuracy: How well a signal type predicted forward returns\n"
            "- strongest_signals: Top N strongest/weakest signals on a date\n"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["symbol_history", "divergence_scan", "regime_scan", "accuracy", "strongest_signals"],
                    "description": "Type of historical query",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Symbols to query (for symbol_history)",
                },
                "trend_regime": {
                    "type": "string",
                    "description": "Filter by trend regime (e.g., 'strong_uptrend', 'downtrend')",
                },
                "divergence_type": {
                    "type": "string",
                    "description": "Filter by divergence type (e.g., 'bullish', 'bearish')",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (ISO format or relative: '3mo', '6mo', '1yr')",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (ISO format or 'now')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 20)",
                },
            },
            "required": ["query_type"],
        },
    }


async def query_ta_history(db, params: dict) -> str:
    """
    Execute a historical TA signal query. Called by Director when Claude
    invokes the query_ta_history tool.
    """
    query_type = params.get("query_type", "symbol_history")
    symbols = params.get("symbols", [])
    trend_regime = params.get("trend_regime")
    divergence_type = params.get("divergence_type")
    start_str = params.get("start_date", "6mo")
    end_str = params.get("end_date", "now")
    limit = min(params.get("limit", 20), 50)

    # Parse dates
    end_date = date.today() if end_str == "now" else _parse_date(end_str, date.today())
    start_date = _parse_relative_date(start_str, end_date)

    parts = [f"TA SIGNAL HISTORY ({start_date} to {end_date}):"]

    try:
        if query_type == "symbol_history":
            result = await _query_symbol_history(db, symbols, start_date, end_date, limit)
            parts.append(result)

        elif query_type == "divergence_scan":
            result = await _query_divergence_scan(db, start_date, end_date, divergence_type, limit)
            parts.append(result)

        elif query_type == "regime_scan":
            result = await _query_regime_scan(db, start_date, end_date, trend_regime, limit)
            parts.append(result)

        elif query_type == "accuracy":
            result = await _query_accuracy(db, start_date, end_date, trend_regime, divergence_type)
            parts.append(result)

        elif query_type == "strongest_signals":
            result = await _query_strongest(db, start_date, end_date, limit)
            parts.append(result)

        else:
            parts.append(f"Unknown query type: {query_type}")

    except Exception as e:
        parts.append(f"ERROR: {e}")

    return "\n".join(parts)


async def _query_symbol_history(db, symbols, start_date, end_date, limit):
    """Get signal history for specific symbols."""
    if not symbols:
        return "ERROR: No symbols specified for symbol_history query."

    rows = await db.fetch(
        """
        SELECT symbol, signal_date, trend_regime, trend_score, momentum_score,
               rsi_14, divergence_type, nearest_support, nearest_resistance,
               fwd_return_5d, fwd_return_20d, close_price
        FROM technical_signals
        WHERE symbol = ANY($1) AND signal_date BETWEEN $2 AND $3
        ORDER BY signal_date DESC
        LIMIT $4
        """,
        symbols, start_date, end_date, limit,
    )

    if not rows:
        return f"No signals found for {', '.join(symbols)} in date range."

    lines = []
    for r in rows:
        fwd = f", 20d return: {float(r['fwd_return_20d'])*100:+.1f}%" if r['fwd_return_20d'] else ""
        div = f", DIV: {r['divergence_type']}" if r['divergence_type'] and r['divergence_type'] != 'none' else ""
        lines.append(
            f"  {r['signal_date']} {r['symbol']}: {r['trend_regime']} "
            f"(trend={float(r['trend_score'] or 0):.2f}, mom={float(r['momentum_score'] or 0):.2f}, "
            f"RSI={float(r['rsi_14'] or 0):.1f}){div}{fwd}"
        )

    return f"Signal history for {', '.join(symbols)}:\n" + "\n".join(lines)


async def _query_divergence_scan(db, start_date, end_date, div_type, limit):
    """Find all divergences in a date range."""
    if div_type:
        rows = await db.fetch(
            """
            SELECT symbol, signal_date, divergence_type, trend_regime, momentum_score,
                   rsi_14, fwd_return_20d, close_price
            FROM technical_signals
            WHERE divergence_type = $1 AND signal_date BETWEEN $2 AND $3
            ORDER BY signal_date DESC
            LIMIT $4
            """,
            div_type, start_date, end_date, limit,
        )
    else:
        rows = await db.fetch(
            """
            SELECT symbol, signal_date, divergence_type, trend_regime, momentum_score,
                   rsi_14, fwd_return_20d, close_price
            FROM technical_signals
            WHERE divergence_type IS NOT NULL AND divergence_type != 'none'
                  AND signal_date BETWEEN $2 AND $3
            ORDER BY signal_date DESC
            LIMIT $4
            """,
            start_date, end_date, limit,
        )

    if not rows:
        return "No divergences found in date range."

    lines = []
    for r in rows:
        fwd = f", 20d: {float(r['fwd_return_20d'])*100:+.1f}%" if r['fwd_return_20d'] else ""
        lines.append(
            f"  {r['signal_date']} {r['symbol']}: {r['divergence_type']} "
            f"(regime={r['trend_regime']}, RSI={float(r['rsi_14'] or 0):.1f}){fwd}"
        )

    return f"Divergences found: {len(rows)}\n" + "\n".join(lines)


async def _query_regime_scan(db, start_date, end_date, regime, limit):
    """Find symbols in a specific trend regime."""
    if not regime:
        return "ERROR: No trend_regime specified for regime_scan."

    # Get the most recent signal date in range
    latest = await db.fetchval(
        "SELECT MAX(signal_date) FROM technical_signals WHERE signal_date BETWEEN $1 AND $2",
        start_date, end_date,
    )
    if not latest:
        return "No signals in date range."

    rows = await db.fetch(
        """
        SELECT symbol, trend_score, momentum_score, rsi_14, divergence_type,
               fwd_return_20d, close_price
        FROM technical_signals
        WHERE trend_regime = $1 AND signal_date = $2
        ORDER BY trend_score DESC
        LIMIT $3
        """,
        regime, latest, limit,
    )

    if not rows:
        return f"No symbols in {regime} on {latest}."

    lines = []
    for r in rows:
        fwd = f", 20d: {float(r['fwd_return_20d'])*100:+.1f}%" if r['fwd_return_20d'] else ""
        lines.append(
            f"  {r['symbol']}: trend={float(r['trend_score'] or 0):.2f}, "
            f"mom={float(r['momentum_score'] or 0):.2f}, RSI={float(r['rsi_14'] or 0):.1f}{fwd}"
        )

    return f"Symbols in {regime} on {latest}: {len(rows)}\n" + "\n".join(lines)


async def _query_accuracy(db, start_date, end_date, regime, div_type):
    """Show signal accuracy statistics."""
    conditions = ["fwd_return_20d IS NOT NULL", "signal_date BETWEEN $1 AND $2"]
    params = [start_date, end_date]

    if regime:
        params.append(regime)
        conditions.append(f"trend_regime = ${len(params)}")
    if div_type:
        params.append(div_type)
        conditions.append(f"divergence_type = ${len(params)}")

    where = " AND ".join(conditions)

    row = await db.fetchrow(
        f"""
        SELECT COUNT(*) as n,
               ROUND((AVG(fwd_return_5d) * 100)::numeric, 2) as avg_5d,
               ROUND((AVG(fwd_return_20d) * 100)::numeric, 2) as avg_20d,
               ROUND((AVG(fwd_return_60d) * 100)::numeric, 2) as avg_60d,
               ROUND((AVG(fwd_max_drawdown_20d) * 100)::numeric, 2) as avg_dd
        FROM technical_signals
        WHERE {where}
        """,
        *params,
    )

    if not row or row["n"] == 0:
        return "No matching signals with forward returns."

    filter_desc = []
    if regime:
        filter_desc.append(f"regime={regime}")
    if div_type:
        filter_desc.append(f"divergence={div_type}")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""

    return (
        f"Signal accuracy{filter_str}:\n"
        f"  Signals: {row['n']:,}\n"
        f"  Avg 5d return: {row['avg_5d']:+.2f}%\n"
        f"  Avg 20d return: {row['avg_20d']:+.2f}%\n"
        f"  Avg 60d return: {row['avg_60d']:+.2f}%\n"
        f"  Avg max drawdown (20d): {row['avg_dd']:+.2f}%"
    )


async def _query_strongest(db, start_date, end_date, limit):
    """Get the strongest bullish and bearish signals."""
    # Get the most recent signal date
    latest = await db.fetchval(
        "SELECT MAX(signal_date) FROM technical_signals WHERE signal_date BETWEEN $1 AND $2",
        start_date, end_date,
    )
    if not latest:
        return "No signals in date range."

    # Top bullish
    bull = await db.fetch(
        """
        SELECT symbol, trend_score, momentum_score, trend_regime, divergence_type
        FROM technical_signals
        WHERE signal_date = $1
        ORDER BY (COALESCE(trend_score, 0) + COALESCE(momentum_score, 0)) DESC
        LIMIT $2
        """,
        latest, limit,
    )

    # Top bearish
    bear = await db.fetch(
        """
        SELECT symbol, trend_score, momentum_score, trend_regime, divergence_type
        FROM technical_signals
        WHERE signal_date = $1
        ORDER BY (COALESCE(trend_score, 0) + COALESCE(momentum_score, 0)) ASC
        LIMIT $2
        """,
        latest, limit,
    )

    lines = [f"Strongest signals on {latest}:"]
    lines.append(f"\n  TOP BULLISH:")
    for r in bull:
        div = f" [{r['divergence_type']}]" if r['divergence_type'] and r['divergence_type'] != 'none' else ""
        lines.append(f"    {r['symbol']}: trend={float(r['trend_score'] or 0):+.2f}, "
                     f"mom={float(r['momentum_score'] or 0):+.2f}, regime={r['trend_regime']}{div}")
    lines.append(f"\n  TOP BEARISH:")
    for r in bear:
        div = f" [{r['divergence_type']}]" if r['divergence_type'] and r['divergence_type'] != 'none' else ""
        lines.append(f"    {r['symbol']}: trend={float(r['trend_score'] or 0):+.2f}, "
                     f"mom={float(r['momentum_score'] or 0):+.2f}, regime={r['trend_regime']}{div}")

    return "\n".join(lines)


def _parse_date(date_str: str, fallback: date) -> date:
    """Parse ISO date string."""
    try:
        parts = date_str.strip().split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return fallback


def _parse_relative_date(date_str: str, reference: date) -> date:
    """Parse relative date like '3mo', '6mo', '1yr'."""
    try:
        s = date_str.strip().lower()
        if s.endswith("mo"):
            months = int(s[:-2])
            return reference - timedelta(days=months * 30)
        elif s.endswith("yr"):
            years = int(s[:-2])
            return reference - timedelta(days=years * 365)
        elif s.endswith("w"):
            weeks = int(s[:-1])
            return reference - timedelta(weeks=weeks)
        else:
            return _parse_date(date_str, reference - timedelta(days=180))
    except Exception:
        return reference - timedelta(days=180)
