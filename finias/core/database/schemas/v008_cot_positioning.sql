-- FINIAS v008: CFTC Commitment of Traders Positioning Data
-- Weekly COT data for 5 futures contracts
-- Source: CFTC via cftc-cot library (official government data)

CREATE TABLE IF NOT EXISTS cot_positioning (
    id              BIGSERIAL PRIMARY KEY,
    contract_key    VARCHAR(20) NOT NULL,       -- sp500, treasury_10y, wti_crude, gold, dollar_index
    contract_name   TEXT NOT NULL,              -- Full CFTC market name
    report_date     DATE NOT NULL,             -- Tuesday date of the COT report
    open_interest   BIGINT,
    noncomm_long    BIGINT,                    -- Noncommercial (speculator) long positions
    noncomm_short   BIGINT,                    -- Noncommercial (speculator) short positions
    net_spec        BIGINT,                    -- noncomm_long - noncomm_short
    fetched_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(contract_key, report_date)          -- One row per contract per week
);

CREATE INDEX IF NOT EXISTS idx_cot_contract_date
    ON cot_positioning(contract_key, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_cot_report_date
    ON cot_positioning(report_date DESC);
