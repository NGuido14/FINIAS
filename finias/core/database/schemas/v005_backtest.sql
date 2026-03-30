-- FINIAS Schema v005: Backtesting Results
-- Stores walk-forward backtest outputs for validation and ML training

CREATE TABLE IF NOT EXISTS backtest_results (
    id                      BIGSERIAL PRIMARY KEY,
    backtest_run_id         UUID NOT NULL,
    sim_date                DATE NOT NULL,

    -- Regime scores from computation pipeline
    composite_score         NUMERIC(6, 4),
    growth_score            NUMERIC(6, 4),
    monetary_score          NUMERIC(6, 4),
    inflation_score         NUMERIC(6, 4),
    market_score            NUMERIC(6, 4),
    primary_regime          VARCHAR(30),
    cycle_phase             VARCHAR(30),
    stress_index            NUMERIC(5, 4),
    confidence              NUMERIC(5, 4),
    binding_constraint      VARCHAR(50),

    -- Forward SPX returns (actual outcomes)
    spx_fwd_5d              NUMERIC(8, 4),
    spx_fwd_20d             NUMERIC(8, 4),
    spx_fwd_60d             NUMERIC(8, 4),
    spx_max_dd_20d          NUMERIC(8, 4),

    -- Metadata
    modules_used            TEXT,
    warmup                  BOOLEAN DEFAULT FALSE,
    created_at              TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(backtest_run_id, sim_date)
);

CREATE INDEX IF NOT EXISTS idx_backtest_run ON backtest_results(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_results(sim_date);
