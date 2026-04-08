"""
FINIAS Symbol Universe — Shared Infrastructure (Layer 0).

Manages the universe of symbols tracked by FINIAS across all agents.
The universe is organized into tiers:

  - MACRO: 19 ETFs used by the Macro Strategist (hardcoded — our specific ETF choices)
  - SP500: ~503 S&P 500 tickers (fetched from Wikipedia at seed time, stored in DB)
  - EXTENDED: Future expansion (mid-cap, watchlist, etc.)

The S&P 500 constituent list is NOT hardcoded. It is fetched from Wikipedia
by the seed_universe.py script and stored in the symbol_universe table.
Helper functions in this module read from that table.

The only hardcoded symbols are the 19 macro ETFs, which are our specific
choice of instruments — not an index that changes quarterly.

Usage:
    from finias.data.universe import (
        MACRO_ETFS,
        get_active_symbols,
        get_sector_for_symbol,
        fetch_sp500_from_wikipedia,
    )
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger("finias.data.universe")


# =============================================================================
# TIER DEFINITIONS
# =============================================================================

TIER_MACRO = "macro"
TIER_SP500 = "sp500"
TIER_EXTENDED = "extended"


# =============================================================================
# MACRO ETFs — Hardcoded (our specific ETF choices for the Macro Strategist)
# =============================================================================

MACRO_ETFS = [
    "SPY", "IWM", "RSP", "TLT", "HYG", "GLD", "CPER", "EEM",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLC", "XLY", "XLRE", "XLB",
]

# Map sector ETFs to their GICS sector name
ETF_SECTOR_MAP = {
    "XLK": "Information Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}


# =============================================================================
# WIKIPEDIA FETCH — Live S&P 500 constituent list
# =============================================================================

def fetch_sp500_from_wikipedia() -> list[dict]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    """
    import pandas as pd
    import io
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 constituents from {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FINIAS/1.0 (Financial Intelligence System)"}
    )
    with urllib.request.urlopen(req) as response:
        html = response.read().decode("utf-8")

    tables = pd.read_html(io.StringIO(html))
    df = tables[0]

    results = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        results.append({
            "symbol": symbol,
            "company_name": str(row.get("Security", "")).strip(),
            "sector": str(row.get("GICS Sector", "")).strip(),
            "sub_industry": str(row.get("GICS Sub-Industry", "")).strip(),
        })

    logger.info(f"Fetched {len(results)} S&P 500 constituents")
    return results


# =============================================================================
# DATABASE HELPERS — Read universe from PostgreSQL
# =============================================================================

async def get_active_symbols(db, tier: Optional[str] = None) -> list[str]:
    """
    Read active symbols from the symbol_universe table.

    Args:
        db: DatabasePool instance.
        tier: Optional filter ('macro', 'sp500', 'extended').
              If None, returns all active symbols.
    """
    if tier:
        rows = await db.fetch(
            "SELECT DISTINCT symbol FROM symbol_universe WHERE is_active AND tier = $1 ORDER BY symbol",
            tier,
        )
    else:
        rows = await db.fetch(
            "SELECT DISTINCT symbol FROM symbol_universe WHERE is_active ORDER BY symbol",
        )
    return [r["symbol"] for r in rows]


async def get_all_tracked_symbols(db) -> list[str]:
    """Return all active symbols from all tiers, deduplicated and sorted."""
    return await get_active_symbols(db, tier=None)


async def get_sector_for_symbol(db, symbol: str) -> Optional[str]:
    """Get GICS sector for a symbol from the database. Falls back to ETF map."""
    row = await db.fetchrow(
        "SELECT sector FROM symbol_universe WHERE symbol = $1 AND is_active LIMIT 1",
        symbol,
    )
    if row and row["sector"]:
        return row["sector"]
    return ETF_SECTOR_MAP.get(symbol)


async def get_symbols_by_sector(db, sector: str) -> list[str]:
    """Get all active symbols in a given GICS sector."""
    rows = await db.fetch(
        "SELECT symbol FROM symbol_universe WHERE sector = $1 AND is_active ORDER BY symbol",
        sector,
    )
    return [r["symbol"] for r in rows]


async def get_sector_names(db) -> list[str]:
    """Return all distinct GICS sector names from active symbols."""
    rows = await db.fetch(
        "SELECT DISTINCT sector FROM symbol_universe WHERE is_active AND sector IS NOT NULL ORDER BY sector",
    )
    return [r["sector"] for r in rows]


async def get_universe_summary(db) -> dict:
    """Get a summary of the universe table for diagnostics."""
    tier_counts = await db.fetch(
        """
        SELECT tier, COUNT(*) as total,
               SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active
        FROM symbol_universe GROUP BY tier ORDER BY tier
        """
    )
    sector_counts = await db.fetch(
        """
        SELECT sector, COUNT(*) as cnt
        FROM symbol_universe WHERE is_active AND tier = 'sp500'
        GROUP BY sector ORDER BY sector
        """
    )
    return {
        "tiers": {r["tier"]: {"total": r["total"], "active": r["active"]} for r in tier_counts},
        "sectors": {r["sector"]: r["cnt"] for r in sector_counts},
    }


async def populate_macro_etfs(db) -> int:
    """
    Populate the symbol_universe table with the 19 macro ETFs.
    Returns count of rows inserted/updated.
    """
    count = 0
    for symbol in MACRO_ETFS:
        sector = ETF_SECTOR_MAP.get(symbol, "Broad Market")
        await db.execute(
            """
            INSERT INTO symbol_universe (symbol, tier, company_name, sector, is_active)
            VALUES ($1, $2, $3, $4, TRUE)
            ON CONFLICT (symbol, tier) DO UPDATE SET
                sector = EXCLUDED.sector, is_active = TRUE
            """,
            symbol, TIER_MACRO, f"{symbol} ETF", sector,
        )
        count += 1
    return count


async def populate_sp500_from_list(db, constituents: list[dict]) -> dict:
    """
    Populate symbol_universe with S&P 500 data from fetch_sp500_from_wikipedia().

    Args:
        db: DatabasePool instance.
        constituents: List of dicts from fetch_sp500_from_wikipedia().

    Returns:
        Summary dict with counts.
    """
    inserted = 0
    for entry in constituents:
        await db.execute(
            """
            INSERT INTO symbol_universe (symbol, tier, company_name, sector, industry, is_active)
            VALUES ($1, $2, $3, $4, $5, TRUE)
            ON CONFLICT (symbol, tier) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                is_active = TRUE
            """,
            entry["symbol"],
            TIER_SP500,
            entry["company_name"],
            entry["sector"],
            entry.get("sub_industry", ""),
        )
        inserted += 1

    return {
        "sp500_count": inserted,
        "macro_count": len(MACRO_ETFS),
    }
