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
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "DGS5": "5-Year Treasury Constant Maturity Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS30": "30-Year Treasury Constant Maturity Rate",
    "DTB3": "3-Month Treasury Bill Secondary Market Rate",

    # Spreads
    "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury (Yield Curve Spread)",
    "T10Y3M": "10-Year Treasury Minus 3-Month Treasury",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",

    # Rates & Policy
    "FEDFUNDS": "Effective Federal Funds Rate",
    "DFEDTARU": "Federal Funds Target Rate Upper Limit",
    "DFEDTARL": "Federal Funds Target Rate Lower Limit",

    # Volatility
    "VIXCLS": "CBOE Volatility Index (VIX)",

    # Inflation
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "T5YIE": "5-Year Breakeven Inflation Rate",
    "T10YIE": "10-Year Breakeven Inflation Rate",

    # Labor
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Claims for Unemployment Insurance",
    "PAYEMS": "All Employees, Total Nonfarm",

    # Sentiment / Activity
    "UMCSENT": "University of Michigan Consumer Sentiment Index",
    "INDPRO": "Industrial Production Index",

    # Dollar
    "DTWEXBGS": "Nominal Broad U.S. Dollar Index",

    # === MONETARY POLICY & LIQUIDITY ===
    "WALCL": "Federal Reserve Total Assets",
    "TREAST": "U.S. Treasury Securities Held by the Federal Reserve",
    "WSHOMCB": "Mortgage-Backed Securities Held by the Federal Reserve",
    "RRPONTSYD": "Overnight Reverse Repurchase Agreements (Treasury Securities Sold)",
    "WTREGEN": "U.S. Treasury General Account (TGA) Balance at Federal Reserve",
    "WRESBAL": "Reserve Balances with Federal Reserve Banks",
    "TOTBKCR": "Bank Credit, All Commercial Banks",
    "TOTALSL": "Total Consumer Credit Owned and Securitized",
    "M2SL": "M2 Money Stock",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "ANFCI": "Chicago Fed Adjusted National Financial Conditions Index",
    "STLFSI4": "St. Louis Fed Financial Stress Index",

    # === BUSINESS CYCLE ===
    # NOTE: USSLIND (Conference Board LEI) was removed from FRED in 2024.
    # Business cycle module handles empty LEI data gracefully.
    "TCU": "Capacity Utilization: Total Industry",
    "PERMIT": "New Privately-Owned Housing Units Authorized by Building Permits",
    "HOUST": "New Privately-Owned Housing Units Started",
    "RSAFS": "Advance Retail Sales: Retail and Food Services",
    "PI": "Personal Income",
    "PCEC96": "Real Personal Consumption Expenditures (Billions of Chained 2017 Dollars)",
    "DGORDER": "Manufacturers' New Orders: Durable Goods",
    "CFNAI": "Chicago Fed National Activity Index",
    "GACDFSA066MSFRBPHI": "Diffusion Index for Future General Activity (Philadelphia Fed Manufacturing)",
    "CCSA": "Continued Claims (Insured Unemployment)",
    "JTSJOL": "Job Openings: Total Nonfarm (JOLTS)",
    "JTSQUR": "Quits Rate: Total Nonfarm (JOLTS)",

    # === INFLATION ===
    "CPILFESL": "Consumer Price Index: All Items Less Food and Energy (Core CPI)",
    "CUSR0000SEHC": "Consumer Price Index: Owners' Equivalent Rent of Residences (Shelter)",
    "CUSR0000SAS": "Consumer Price Index: Services",
    "PCEPI": "Personal Consumption Expenditures: Chain-type Price Index",
    "PCEPILFE": "Personal Consumption Expenditures Excluding Food and Energy (Core PCE)",
    "STICKCPIM157SFRBATL": "Sticky Price Consumer Price Index (Atlanta Fed)",
    "FLEXCPIM157SFRBATL": "Flexible Price Consumer Price Index (Atlanta Fed)",
    "PCETRIM12M159SFRBDAL": "Trimmed Mean PCE Inflation Rate (Dallas Fed)",
    "T5YIFR": "5-Year, 5-Year Forward Inflation Expectation Rate",
    "PPIACO": "Producer Price Index: All Commodities",
    "CES0500000003": "Average Hourly Earnings of All Employees, Total Private",
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI)",

    # === YIELD CURVE ENHANCEMENTS ===
    "DFII5": "5-Year Treasury Inflation-Indexed Security (TIPS Real Yield)",
    "DFII10": "10-Year Treasury Inflation-Indexed Security (TIPS Real Yield)",
    "THREEFYTP10": "10-Year Treasury Term Premium (Adrian-Crump-Moench)",

    # === LABOR MARKET ===
    "U6RATE": "Total Unemployed Plus Marginally Attached Plus Part-Time for Economic Reasons (U-6)",
    "CIVPART": "Civilian Labor Force Participation Rate",
    "LNS11300060": "Employment-Population Ratio: 25-54 Years (Prime Age)",
    "TEMPHELPS": "All Employees: Temporary Help Services",
    "AWHAETP": "Average Weekly Hours of All Employees, Total Private",
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
            "sort_order": "asc",
        }
        if observation_start:
            params["observation_start"] = observation_start.isoformat()
        if observation_end:
            params["observation_end"] = observation_end.isoformat()

        url = f"{self.BASE_URL}/series/observations"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"FRED API returned {resp.status} for {series_id}")
                return []
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
            return observations[-1]  # Sorted ascending, latest is last
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
