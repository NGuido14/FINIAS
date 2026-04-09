-- FINIAS Schema v013: Add 1-day forward return for short-term signal validation
ALTER TABLE technical_signals
    ADD COLUMN IF NOT EXISTS fwd_return_1d NUMERIC(8, 4);
