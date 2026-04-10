-- FINIAS Schema v014: Richer Macro Context for Technical Signals
-- Adds multi-dimensional macro context beyond the single regime label.
-- These fields have real discriminating power for TA signal quality:
--   binding_constraint: 3 values (inflation 73%, growth 22%, monetary 5%)
--   volatility_regime: 4 values, changes during crises
--   cycle_phase: 4 values, drives sector rotation
--   composite_score: continuous, more granular than regime label
--   stress_index: predicts drawdowns (correlation -0.256)

ALTER TABLE technical_signals
    ADD COLUMN IF NOT EXISTS macro_binding VARCHAR(30),
    ADD COLUMN IF NOT EXISTS macro_volatility VARCHAR(20),
    ADD COLUMN IF NOT EXISTS macro_cycle_phase VARCHAR(20),
    ADD COLUMN IF NOT EXISTS macro_composite NUMERIC(6, 4),
    ADD COLUMN IF NOT EXISTS macro_stress NUMERIC(6, 4);

-- Index for cross-referencing TA signals with macro context
CREATE INDEX IF NOT EXISTS idx_ta_signals_macro_binding
    ON technical_signals(macro_binding, signal_date DESC)
    WHERE macro_binding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ta_signals_macro_stress
    ON technical_signals(macro_stress)
    WHERE macro_stress IS NOT NULL;
