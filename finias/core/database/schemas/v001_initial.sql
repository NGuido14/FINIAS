-- FINIAS Initial Schema
-- v001: Core tables for Sprint 0 (Director + Macro Strategist)

-- =============================================================================
-- MARKET DATA
-- =============================================================================

-- Daily OHLCV price data from Polygon
CREATE TABLE IF NOT EXISTS market_data_daily (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    trade_date      DATE NOT NULL,
    open            NUMERIC(14, 4),
    high            NUMERIC(14, 4),
    low             NUMERIC(14, 4),
    close           NUMERIC(14, 4),
    volume          BIGINT,
    vwap            NUMERIC(14, 4),
    num_trades      INTEGER,
    source          VARCHAR(50) DEFAULT 'polygon',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date
    ON market_data_daily(symbol, trade_date DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_date
    ON market_data_daily(trade_date DESC);


-- Economic indicator data from FRED
CREATE TABLE IF NOT EXISTS economic_indicators (
    id              BIGSERIAL PRIMARY KEY,
    series_id       VARCHAR(50) NOT NULL,      -- FRED series ID (e.g., 'DGS10', 'VIXCLS')
    obs_date        DATE NOT NULL,
    value           NUMERIC(18, 6),
    series_name     VARCHAR(255),              -- Human-readable name
    frequency       VARCHAR(20),               -- daily, weekly, monthly
    units           VARCHAR(100),              -- percent, index, etc.
    source          VARCHAR(50) DEFAULT 'fred',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(series_id, obs_date)
);

CREATE INDEX IF NOT EXISTS idx_econ_series_date
    ON economic_indicators(series_id, obs_date DESC);


-- =============================================================================
-- AGENT OPINIONS
-- =============================================================================

-- Every opinion from every agent is stored for history and audit
CREATE TABLE IF NOT EXISTS agent_opinions (
    id              BIGSERIAL PRIMARY KEY,
    opinion_id      UUID NOT NULL UNIQUE,
    agent_name      VARCHAR(100) NOT NULL,
    agent_layer     INTEGER NOT NULL,

    -- Assessment
    direction       VARCHAR(30) NOT NULL,
    confidence      VARCHAR(20) NOT NULL,
    regime          VARCHAR(30),

    -- Content
    summary         TEXT NOT NULL,
    key_findings    JSONB NOT NULL DEFAULT '[]',
    data_points     JSONB NOT NULL DEFAULT '{}',
    methodology     TEXT,
    risks_to_view   JSONB NOT NULL DEFAULT '[]',
    watch_items     JSONB NOT NULL DEFAULT '[]',

    -- Metadata
    data_freshness  TIMESTAMPTZ,
    computation_ms  NUMERIC(10, 2),

    -- Query context
    query_question  TEXT,
    query_context   JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_opinions_agent_time
    ON agent_opinions(agent_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_opinions_regime
    ON agent_opinions(regime, created_at DESC);


-- =============================================================================
-- MACRO REGIME HISTORY
-- =============================================================================

-- Tracks regime changes over time for the Macro Strategist
CREATE TABLE IF NOT EXISTS regime_history (
    id              BIGSERIAL PRIMARY KEY,
    regime          VARCHAR(30) NOT NULL,
    confidence      NUMERIC(5, 4) NOT NULL,    -- 0.0000 to 1.0000

    -- Component scores that drove the regime assessment
    vix_level       NUMERIC(8, 4),
    vix_regime      VARCHAR(30),
    yield_curve_2s10s   NUMERIC(8, 4),
    yield_curve_3m10y   NUMERIC(8, 4),
    credit_spread       NUMERIC(8, 4),
    breadth_pct_above_200ma  NUMERIC(6, 4),
    adv_decline_ratio        NUMERIC(8, 4),
    dxy_level               NUMERIC(8, 4),

    -- Composite scores
    stress_index        NUMERIC(6, 4),         -- 0-1 composite stress
    momentum_score      NUMERIC(6, 4),         -- -1 to 1 momentum

    assessed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_time
    ON regime_history(assessed_at DESC);


-- =============================================================================
-- SYSTEM METADATA
-- =============================================================================

-- Track schema migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(20) PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

-- Agent health tracking
CREATE TABLE IF NOT EXISTS agent_health_log (
    id              BIGSERIAL PRIMARY KEY,
    agent_name      VARCHAR(100) NOT NULL,
    is_healthy      BOOLEAN NOT NULL,
    data_freshness  TIMESTAMPTZ,
    error_message   TEXT,
    details         JSONB DEFAULT '{}',
    checked_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_health_agent_time
    ON agent_health_log(agent_name, checked_at DESC);
