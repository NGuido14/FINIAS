"""
Mapping between FRED series IDs and macro_data_matrix column names.

This is the single source of truth for how FRED series map to matrix columns.
"""

# FRED series_id → matrix column name
SERIES_TO_COLUMN = {
    # Yields
    "DGS2": "dgs2",
    "DGS5": "dgs5",
    "DGS10": "dgs10",
    "DGS30": "dgs30",
    "DTB3": "dtb3",

    # Spreads
    "T10Y2Y": "t10y2y",
    "T10Y3M": "t10y3m",
    "BAMLH0A0HYM2": "hy_oas",

    # Rates
    "FEDFUNDS": "fedfunds",
    "DFEDTARU": "fed_target_upper",
    "DFEDTARL": "fed_target_lower",

    # Volatility
    "VIXCLS": "vix",

    # Fed Balance Sheet & Liquidity
    "WALCL": "fed_total_assets",
    "TREAST": "fed_treasuries",
    "WSHOMCB": "fed_mbs",
    "RRPONTSYD": "reverse_repo",
    "WTREGEN": "tga_balance",
    "WRESBAL": "bank_reserves",

    # Credit & Money
    "TOTBKCR": "bank_credit",
    "TOTALSL": "consumer_credit",
    "M2SL": "m2",

    # Financial Conditions
    "NFCI": "nfci",
    "ANFCI": "anfci",
    "STLFSI4": "stlfsi",

    # Inflation
    "CPIAUCSL": "cpi_all",
    "CPILFESL": "cpi_core",
    "CUSR0000SEHC": "cpi_shelter",
    "CUSR0000SAS": "cpi_services",
    "PCEPI": "pce",
    "PCEPILFE": "core_pce",
    "STICKCPIM159SFRBATL": "sticky_cpi",
    "FLEXCPIM159SFRBATL": "flexible_cpi",
    "PCETRIM12M159SFRBDAL": "trimmed_mean_pce",
    "PPIACO": "ppi_all",
    "DCOILWTICO": "oil_wti",
    "DCOILBRENTEU": "oil_brent",

    # Breakevens & Real Yields
    "T5YIE": "breakeven_5y",
    "T10YIE": "breakeven_10y",
    "T5YIFR": "forward_5y5y",
    "DFII5": "tips_5y",
    "DFII10": "tips_10y",
    "THREEFYTP10": "term_premium_10y",

    # Labor
    "UNRATE": "unemployment",
    "U6RATE": "u6_unemployment",
    "ICSA": "initial_claims",
    "CCSA": "continuing_claims",
    "PAYEMS": "nonfarm_payrolls",
    "JTSJOL": "jolts_openings",
    "JTSQUR": "jolts_quits_rate",
    "CIVPART": "participation_rate",
    "LNS11300060": "prime_age_epop",
    "TEMPHELPS": "temp_employment",
    "AWHAETP": "avg_weekly_hours",
    "CES0500000003": "avg_hourly_earnings",

    # Activity & Sentiment
    # PHLEAD/USSLIND (LEI) removed — discontinued on FRED
    "INDPRO": "industrial_production",
    "TCU": "capacity_utilization",
    "PERMIT": "building_permits",
    "HOUST": "housing_starts",
    "RSAFS": "retail_sales",
    "PI": "personal_income",
    "PCEC96": "personal_consumption",
    "DGORDER": "durable_goods",
    "CFNAI": "cfnai",
    "GACDFSA066MSFRBPHI": "philly_fed",
    "UMCSENT": "consumer_sentiment",

    # Dollar
    "DTWEXBGS": "dxy",
}

# Reverse mapping: column name → FRED series_id
COLUMN_TO_SERIES = {v: k for k, v in SERIES_TO_COLUMN.items()}
