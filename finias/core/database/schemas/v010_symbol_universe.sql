-- FINIAS Schema v010: Symbol Universe
-- Tracks all symbols monitored by FINIAS across agent tiers.
-- Used by Technical Analyst, Fundamental Analyst, Screening Agent, and others.

CREATE TABLE IF NOT EXISTS symbol_universe (
    symbol          VARCHAR(20) NOT NULL,
    tier            VARCHAR(20) NOT NULL,           -- macro, sp500, extended
    company_name    VARCHAR(255),
    sector          VARCHAR(100),                   -- GICS sector
    industry        VARCHAR(255),                   -- GICS sub-industry
    market_cap_tier VARCHAR(20),                    -- mega, large, mid, small (future)
    added_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    removed_date    DATE,                           -- NULL = still active
    is_active       BOOLEAN DEFAULT TRUE,
    metadata        JSONB DEFAULT '{}',

    PRIMARY KEY (symbol, tier)
);

-- Fast lookups by tier + active status (primary query pattern for agents)
CREATE INDEX IF NOT EXISTS idx_universe_tier_active
    ON symbol_universe(tier, is_active) WHERE is_active;

-- Sector-based queries for TA sector analysis and Fundamental peer groups
CREATE INDEX IF NOT EXISTS idx_universe_sector
    ON symbol_universe(sector) WHERE is_active;

-- Removed symbols lookup for survivorship bias tracking
CREATE INDEX IF NOT EXISTS idx_universe_removed
    ON symbol_universe(removed_date) WHERE removed_date IS NOT NULL;
