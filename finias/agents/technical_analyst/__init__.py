"""
FINIAS Technical Analyst Agent

Layer 1 Domain Expert: Price-based technical analysis across the S&P 500 universe.

Pure Python computation using pandas-ta — no Claude API calls, $0.00 per run.
Reads macro context from Redis to condition signal reliability.

Computation modules:
  - trend.py: Multi-timeframe trend analysis (Ichimoku, ADX, MA constellation)
  - momentum.py: Regime-adaptive momentum (RSI, MACD, Stochastic, divergences)
  - levels.py: Support/resistance (pivots, Bollinger, key level clustering)
  - volume.py: Volume confirmation (OBV, A/D, MFI, relative volume) [Prompt 3]
  - ta_volatility.py: Squeeze detection, ATR, Supertrend [Prompt 3]
  - relative_strength.py: Intermarket analysis, RS rankings [Prompt 3]
  - signals.py: 4-dimension confluence synthesis with macro overlay [Prompt 4]
"""
