from __future__ import annotations
from typing import Any, Optional
from datetime import date, datetime, timezone
import asyncio
import logging
import aiohttp

from finias.core.config.settings import get_settings

logger = logging.getLogger("finias.data.fred")


# Key FRED series IDs used by the Macro Strategist
MACRO_SERIES = {
    # Treasury Yields
    "DGS2": "2-Year Treasury Yield",
    "DGS5": "5-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "DGS30": "30-Year Treasury Yield",
    "DTB3": "3-Month Treasury Bill",

    # Spreads
    "T10Y2Y": "10Y-2Y Spread (Yield Curve)",
    "T10Y3M": "10Y-3M Spread",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS (Credit Spread)",

    # Rates & Policy
    "FEDFUNDS": "Federal Funds Rate",
    "DFEDTARU": "Fed Funds Target Upper",
    "DFEDTARL": "Fed Funds Target Lower",

    # Volatility
    "VIXCLS": "CBOE VIX",

    # Inflation
    "CPIAUCSL": "CPI All Urban Consumers",
    "T5YIE": "5-Year Breakeven Inflation",
    "T10YIE": "10-Year Breakeven Inflation",

    # Labor
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Jobless Claims",
    "PAYEMS": "Total Nonfarm Payrolls",

    # Sentiment / Activity
    "UMCSENT": "U of Michigan Consumer Sentiment",
    "INDPRO": "Industrial Production Index",

    # Dollar
    "DTWEXBGS": "Trade Weighted US Dollar Index (Broad Goods & Services)",

    # === MONETARY POLICY & LIQUIDITY (Phase 1 Expansion) ===
    "WALCL": "Federal Reserve Total Assets",
    "TREAST": "Treasury General Account",
    "WSHOMCB": "Fed Swap Lines with Other Central Banks",
    "RRPONTSYD": "Reverse Repo Operations (Overnight)",
    "WTREGEN": "Treasury Repo Operations (Fed)",
    "WRESBAL": "Fed Reserve Balance (Final)",
    "TOTBKCR": "Total Banking Credit",
    "TOTALSL": "Total Federal Reserve Credit",
    "M2SL": "M2 Money Supply",
    "NFCI": "National Financial Conditions Index",
    "ANFCI": "Adjusted NFCI",
    "STLFSI2": "St. Louis Fed Financial Stress Index",

    # === BUSINESS CYCLE (Phase 1 Expansion) ===
    "USSLIND": "Leading Economic Index",
    "TCU": "Total Capacity Utilization",
    "PERMIT": "New Private Housing Permits",
    "HOUST": "Total Housing Starts",
    "RSAFS": "Retail Sales (Advance)",
    "PI": "Personal Income",
    "PCE": "Personal Consumption Expenditures",
    "DGORDER": "Durable Goods New Orders",
    "CFNAI": "Chicago Fed National Activity Index",
    "GACDISA066MSFRBPHI": "Coincident Economic Activity Index (Philadelphia Fed)",
    "CCSA": "Consumer Credit (Senior Loan Officer Survey)",
    "JTSJOL": "Job Openings",
    "JTSQUR": "Quits Rate",

    # === INFLATION (Phase 1 Expansion) ===
    "CPILFESL": "Core CPI (Less Food & Energy)",
    "CUSR0000SEHC": "Shelter CPI",
    "CUSR0000SAS": "Supercore CPI (Shelter + Core Services)",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE (Less Food & Energy)",
    "STICKCPIM157SFRBATL": "Sticky CPI (Atlanta Fed)",
    "FLEXCPIM157SFRBATL": "Flexible CPI (Atlanta Fed)",
    "PCETRIM12M159SFRBDAL": "Trimmed Mean PCE (Dallas Fed)",
    "T5YIFR": "5-Year Forward Inflation Expectation Rate",
    "PPIACO": "Producer Price Index (All Commodities)",
    "CES0500000003": "Average Hourly Earnings (All Employees)",
    "DCOILWTICO": "Crude Oil WTI Spot Price",

    # === YIELD CURVE ENHANCEMENTS (Phase 1 Expansion) ===
    "DFII5": "5-Year Implied Inflation Rate",
    "DFII10": "10-Year Implied Inflation Rate",
    "THREEFYTP10": "3-Year Forward 10-Year Inflation Rate",

    # === LABOR MARKET (Phase 1 Expansion) ===
    "U6RATE": "Alternative Unemployment Rate (U-6)",
    "CIVPART": "Civilian Labor Force Participation Rate",
    "LNS11300060": "Employees on Nonfarm Payroll (Private Sector)",
    "TEMPHELPS": "Temporary Help Services Employment",
    "AWHAETP": "Average Weekly Hours (Private Sector)",
}


class FredClient:
    """
    Async FRED API client for economic data.

    FRED rate limit: 120 requests per minute.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_settings().fred_api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_series(
        self,
        series_id: str,
        observation_start: Optional[date] = None,
        observation_end: Optional[date] = None
    ) -> list[dict[str, Any]]:
        """
        Get observations for a FRED series.

        Returns list of dicts with keys: date, value
        Missing values (indicated by ".") are filtered out.
        """
        session = await self._get_session()
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
        }
        if observation_start:
            params["observation_start"] = observation_start.isoformat()
        if observation_end:
            params["observation_end"] = observation_end.isoformat()

        url = f"{self.BASE_URL}/series/observations"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        observations = []
        for obs in data.get("observations", []):
            if obs["value"] != ".":
                observations.append({
                    "date": obs["date"],
                    "value": float(obs["value"])
                })

        return observations

    async def get_latest_value(self, series_id: str) -> Optional[dict[str, Any]]:
        """Get the most recent observation for a series."""
        observations = await self.get_series(series_id)
        if observations:
            return observations[0]  # Already sorted desc
        return None

    async def get_macro_snapshot(
        self,
        lookback_days: int = 365
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch all key macro series used by the Macro Strategist.

        Returns dict mapping series_id to list of observations.
        """
        from_date = date.today() - __import__("datetime").timedelta(days=lookback_days)

        results = {}
        for series_id, name in MACRO_SERIES.items():
            try:
                obs = await self.get_series(series_id, observation_start=from_date)
                results[series_id] = obs
                logger.debug(f"Fetched {series_id} ({name}): {len(obs)} observations")
            except Exception as e:
                logger.error(f"Failed to fetch {series_id} ({name}): {e}")
                results[series_id] = []

            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)

        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
