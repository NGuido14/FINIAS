-- FINIAS Schema v012: Forward Returns for Technical Signals
-- Adds actual forward return columns to technical_signals for prediction tracking.
-- These are filled AFTER the fact — when we know what actually happened.

ALTER TABLE technical_signals
    ADD COLUMN IF NOT EXISTS fwd_return_5d NUMERIC(8, 4),
    ADD COLUMN IF NOT EXISTS fwd_return_20d NUMERIC(8, 4),
    ADD COLUMN IF NOT EXISTS fwd_return_60d NUMERIC(8, 4),
    ADD COLUMN IF NOT EXISTS fwd_max_drawdown_20d NUMERIC(8, 4),
    ADD COLUMN IF NOT EXISTS close_price NUMERIC(14, 4);

-- Index for accuracy queries (filter by signal type, measure returns)
CREATE INDEX IF NOT EXISTS idx_ta_signals_accuracy
    ON technical_signals(trend_regime, divergence_type)
    WHERE fwd_return_20d IS NOT NULL;
