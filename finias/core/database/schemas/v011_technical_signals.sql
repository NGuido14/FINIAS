-- FINIAS Schema v011: Technical Signals
-- Persists TA agent computation output for prediction tracking and audit trail.

CREATE TABLE IF NOT EXISTS technical_signals (
    id                  BIGSERIAL PRIMARY KEY,
    symbol              VARCHAR(20) NOT NULL,
    signal_date         DATE NOT NULL,

    -- Trend (Dimension 1)
    trend_regime        VARCHAR(30),
    trend_score         NUMERIC(6, 4),
    adx                 NUMERIC(6, 2),
    ma_alignment        VARCHAR(20),
    ichimoku_signal     VARCHAR(20),
    trend_maturity      VARCHAR(20),

    -- Momentum (Dimension 2)
    momentum_score      NUMERIC(6, 4),
    rsi_14              NUMERIC(6, 2),
    rsi_zone            VARCHAR(20),
    macd_direction      VARCHAR(20),
    macd_cross          VARCHAR(20),
    divergence_type     VARCHAR(30),

    -- Levels
    nearest_support     NUMERIC(14, 4),
    nearest_resistance  NUMERIC(14, 4),
    risk_reward_ratio   NUMERIC(6, 2),

    -- Full computation output (JSONB for complete detail)
    full_signals_json   JSONB DEFAULT '{}',

    -- Metadata
    macro_regime        VARCHAR(30),
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(symbol, signal_date)
);

CREATE INDEX IF NOT EXISTS idx_ta_signals_symbol_date
    ON technical_signals(symbol, signal_date DESC);

CREATE INDEX IF NOT EXISTS idx_ta_signals_date
    ON technical_signals(signal_date DESC);

CREATE INDEX IF NOT EXISTS idx_ta_signals_trend
    ON technical_signals(trend_regime, signal_date DESC);

CREATE INDEX IF NOT EXISTS idx_ta_signals_divergence
    ON technical_signals(divergence_type, signal_date DESC)
    WHERE divergence_type != 'none';
