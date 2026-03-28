-- FINIAS Phase 1 Schema Extension
-- v002: Expanded Macro Strategist tables

-- =============================================================================
-- EXPANDED REGIME HISTORY (replaces simple regime tracking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS regime_assessments (
    id                  BIGSERIAL PRIMARY KEY,

    -- Multi-dimensional regime
    primary_regime      VARCHAR(30) NOT NULL,
    cycle_phase         VARCHAR(30),
    liquidity_regime    VARCHAR(30),
    volatility_regime   VARCHAR(30),
    inflation_regime    VARCHAR(30),

    -- Category scores (-1 to +1)
    growth_cycle_score      NUMERIC(6, 4),
    monetary_liquidity_score NUMERIC(6, 4),
    inflation_score         NUMERIC(6, 4),
    market_signals_score    NUMERIC(6, 4),

    -- Dynamic weights (sum to 1.0)
    weight_growth           NUMERIC(5, 4),
    weight_monetary         NUMERIC(5, 4),
    weight_inflation        NUMERIC(5, 4),
    weight_market           NUMERIC(5, 4),

    -- Composite
    composite_score         NUMERIC(6, 4),
    confidence              NUMERIC(5, 4),
    stress_index            NUMERIC(5, 4),
    binding_constraint      VARCHAR(50),

    -- Key levels for quick reference
    vix_level               NUMERIC(8, 4),
    fed_funds_rate          NUMERIC(6, 4),
    net_liquidity_trillion  NUMERIC(8, 4),
    spread_2s10s            NUMERIC(8, 4),
    core_pce_yoy            NUMERIC(6, 4),
    unemployment_rate       NUMERIC(6, 4),
    sahm_value              NUMERIC(6, 4),
    ism_manufacturing       NUMERIC(6, 4),
    hy_spread               NUMERIC(8, 4),
    nfci                    NUMERIC(8, 4),

    assessed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_assessments_time
    ON regime_assessments(assessed_at DESC);

-- =============================================================================
-- BUSINESS CYCLE TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS cycle_phase_history (
    id                  BIGSERIAL PRIMARY KEY,
    cycle_phase         VARCHAR(30) NOT NULL,
    phase_confidence    NUMERIC(5, 4),

    lei_level           NUMERIC(10, 4),
    lei_mom_change      NUMERIC(8, 4),
    composite_leading   NUMERIC(6, 4),
    sahm_value          NUMERIC(6, 4),
    sahm_triggered      BOOLEAN,
    recession_probability NUMERIC(5, 4),
    ism_manufacturing   NUMERIC(6, 4),
    gdp_nowcast         NUMERIC(6, 4),

    assessed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cycle_phase_time
    ON cycle_phase_history(assessed_at DESC);

-- =============================================================================
-- MONETARY POLICY TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS monetary_policy_history (
    id                  BIGSERIAL PRIMARY KEY,

    fed_funds_rate      NUMERIC(6, 4),
    net_liquidity       NUMERIC(14, 2),
    net_liquidity_change_30d NUMERIC(10, 2),
    fed_balance_sheet   NUMERIC(14, 2),
    tga_level           NUMERIC(14, 2),
    reverse_repo        NUMERIC(14, 2),

    policy_stance       VARCHAR(30),
    liquidity_regime    VARCHAR(30),
    policy_score        NUMERIC(6, 4),
    liquidity_score     NUMERIC(6, 4),

    assessed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_monetary_policy_time
    ON monetary_policy_history(assessed_at DESC);

-- =============================================================================
-- INFLATION TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS inflation_history (
    id                  BIGSERIAL PRIMARY KEY,

    cpi_yoy             NUMERIC(6, 4),
    core_cpi_yoy        NUMERIC(6, 4),
    core_cpi_3m_ann     NUMERIC(6, 4),
    core_pce_yoy        NUMERIC(6, 4),
    supercore_yoy       NUMERIC(6, 4),
    shelter_yoy         NUMERIC(6, 4),

    breakeven_5y        NUMERIC(6, 4),
    breakeven_10y       NUMERIC(6, 4),
    forward_5y5y        NUMERIC(6, 4),
    expectations_anchored BOOLEAN,

    sticky_cpi_yoy      NUMERIC(6, 4),
    flexible_cpi_yoy    NUMERIC(6, 4),
    ahe_yoy             NUMERIC(6, 4),

    inflation_regime    VARCHAR(30),
    inflation_trend     VARCHAR(30),
    spiral_risk         NUMERIC(5, 4),

    assessed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inflation_time
    ON inflation_history(assessed_at DESC);
