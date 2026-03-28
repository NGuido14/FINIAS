-- FINIAS Schema v003: Macro Data Matrix
-- One row per date, one column per FRED series.
-- Populated from economic_indicators after data fetch.
-- Enables single-query access to all macro data, date-aligned by definition.

CREATE TABLE IF NOT EXISTS macro_data_matrix (
    obs_date                DATE PRIMARY KEY,

    -- === TREASURY YIELDS ===
    dgs2                    NUMERIC(10, 4),    -- 2-Year Treasury
    dgs5                    NUMERIC(10, 4),    -- 5-Year Treasury
    dgs10                   NUMERIC(10, 4),    -- 10-Year Treasury
    dgs30                   NUMERIC(10, 4),    -- 30-Year Treasury
    dtb3                    NUMERIC(10, 4),    -- 3-Month T-Bill

    -- === SPREADS ===
    t10y2y                  NUMERIC(10, 4),    -- 10Y-2Y Spread
    t10y3m                  NUMERIC(10, 4),    -- 10Y-3M Spread
    hy_oas                  NUMERIC(10, 4),    -- High Yield OAS (BAMLH0A0HYM2)

    -- === RATES & POLICY ===
    fedfunds                NUMERIC(10, 4),    -- Effective Fed Funds Rate
    fed_target_upper        NUMERIC(10, 4),    -- DFEDTARU
    fed_target_lower        NUMERIC(10, 4),    -- DFEDTARL

    -- === VOLATILITY ===
    vix                     NUMERIC(10, 4),    -- CBOE VIX

    -- === FED BALANCE SHEET & LIQUIDITY ===
    fed_total_assets        NUMERIC(16, 2),    -- WALCL (millions)
    fed_treasuries          NUMERIC(16, 2),    -- TREAST (millions)
    fed_mbs                 NUMERIC(16, 2),    -- WSHOMCB (millions)
    reverse_repo            NUMERIC(16, 2),    -- RRPONTSYD (millions)
    tga_balance             NUMERIC(16, 2),    -- WTREGEN (millions)
    bank_reserves           NUMERIC(16, 2),    -- WRESBAL (millions)
    net_liquidity           NUMERIC(16, 2),    -- Computed: WALCL - WTREGEN - RRPONTSYD

    -- === CREDIT & MONEY SUPPLY ===
    bank_credit             NUMERIC(16, 2),    -- TOTBKCR
    consumer_credit         NUMERIC(16, 2),    -- TOTALSL
    m2                      NUMERIC(16, 2),    -- M2SL

    -- === FINANCIAL CONDITIONS ===
    nfci                    NUMERIC(10, 4),    -- Chicago Fed NFCI
    anfci                   NUMERIC(10, 4),    -- Adjusted NFCI
    stlfsi                  NUMERIC(10, 4),    -- St. Louis Financial Stress

    -- === INFLATION ===
    cpi_all                 NUMERIC(12, 4),    -- CPIAUCSL
    cpi_core                NUMERIC(12, 4),    -- CPILFESL
    cpi_shelter             NUMERIC(12, 4),    -- CUSR0000SEHC
    cpi_services            NUMERIC(12, 4),    -- CUSR0000SAS
    pce                     NUMERIC(12, 4),    -- PCEPI
    core_pce                NUMERIC(12, 4),    -- PCEPILFE
    sticky_cpi              NUMERIC(10, 4),    -- STICKCPIM157SFRBATL
    flexible_cpi            NUMERIC(10, 4),    -- FLEXCPIM157SFRBATL
    trimmed_mean_pce        NUMERIC(10, 4),    -- PCETRIM12M159SFRBDAL
    ppi_all                 NUMERIC(12, 4),    -- PPIACO
    oil_wti                 NUMERIC(10, 4),    -- DCOILWTICO

    -- === BREAKEVENS & REAL YIELDS ===
    breakeven_5y            NUMERIC(10, 4),    -- T5YIE
    breakeven_10y           NUMERIC(10, 4),    -- T10YIE
    forward_5y5y            NUMERIC(10, 4),    -- T5YIFR
    tips_5y                 NUMERIC(10, 4),    -- DFII5
    tips_10y                NUMERIC(10, 4),    -- DFII10
    term_premium_10y        NUMERIC(10, 4),    -- THREEFYTP10

    -- === LABOR MARKET ===
    unemployment            NUMERIC(8, 4),     -- UNRATE
    u6_unemployment         NUMERIC(8, 4),     -- U6RATE
    initial_claims          NUMERIC(12, 2),    -- ICSA
    continuing_claims       NUMERIC(12, 2),    -- CCSA
    nonfarm_payrolls        NUMERIC(12, 2),    -- PAYEMS
    jolts_openings          NUMERIC(12, 2),    -- JTSJOL
    jolts_quits_rate        NUMERIC(8, 4),     -- JTSQUR
    participation_rate      NUMERIC(8, 4),     -- CIVPART
    prime_age_epop          NUMERIC(8, 4),     -- LNS11300060
    temp_employment         NUMERIC(12, 2),    -- TEMPHELPS
    avg_weekly_hours        NUMERIC(8, 4),     -- AWHAETP
    avg_hourly_earnings     NUMERIC(10, 4),    -- CES0500000003

    -- === ACTIVITY & SENTIMENT ===
    lei                     NUMERIC(10, 4),    -- USSLIND
    industrial_production   NUMERIC(10, 4),    -- INDPRO
    capacity_utilization    NUMERIC(8, 4),     -- TCU
    building_permits        NUMERIC(12, 2),    -- PERMIT
    housing_starts          NUMERIC(12, 2),    -- HOUST
    retail_sales            NUMERIC(14, 2),    -- RSAFS
    personal_income         NUMERIC(14, 2),    -- PI
    personal_consumption    NUMERIC(14, 2),    -- PCE (spending)
    durable_goods           NUMERIC(14, 2),    -- DGORDER
    cfnai                   NUMERIC(10, 4),    -- CFNAI
    philly_fed              NUMERIC(10, 4),    -- GACDISA066MSFRBPHI
    consumer_sentiment      NUMERIC(10, 4),    -- UMCSENT

    -- === DOLLAR ===
    dxy                     NUMERIC(10, 4),    -- DTWEXBGS

    -- Metadata
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_macro_matrix_date
    ON macro_data_matrix(obs_date DESC);
