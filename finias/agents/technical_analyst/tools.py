"""
Technical Analyst tool definitions.

Exposes the TA agent as a tool for the Director via Claude's tool_use.
"""


def get_ta_tool_definition() -> dict:
    """
    Claude tool_use definition for the Technical Analyst.

    This tells the Director's Claude instance what the TA agent can do
    and what inputs it accepts. Be VERY specific — Claude uses this
    description to decide when to route queries to this agent.
    """
    return {
        "name": "query_technical_analyst",
        "description": (
            "Query the Technical Analyst for price-based analysis using 7 computation "
            "modules across ~500 S&P 500 stocks. By default analyzes 19 major ETFs for "
            "fast response. To analyze specific stocks, pass their tickers in the "
            "'symbols' parameter. All computation is pure Python — precise, deterministic, "
            "$0.00 per run. Results are NOT LLM-generated.\n\n"
            "7 COMPUTATION MODULES:\n"
            "(1) Trend Analysis — Ichimoku Cloud, ADX trend strength, SMA/EMA constellation "
            "(8/21/50/200), trend regime classification (strong_uptrend → strong_downtrend), "
            "trend maturity, higher-highs/higher-lows structure detection.\n"
            "(2) Regime-Adaptive Momentum — RSI with adaptive thresholds (shift based on "
            "trend regime), MACD direction/acceleration/crosses, Stochastic, Rate of Change, "
            "momentum divergence detection (regular + hidden), momentum thrust.\n"
            "(3) Support & Resistance — Classic/Fibonacci pivots, Bollinger Bands, Donchian "
            "channels, key level clustering, distance to nearest S/R, risk/reward ratio.\n"
            "(4) Volume Confirmation — OBV trend + divergence (institutional behavior), "
            "relative volume vs 20d average, volume trend during regime, MFI, A/D line. "
            "Scores whether volume confirms or contradicts the price move.\n"
            "(5) Relative Strength — Stock vs sector RS ratio, sector vs SPY, RS percentile "
            "across universe, RS momentum (improving/deteriorating), RS regime classification "
            "(leading, improving, lagging, deteriorating).\n"
            "(6) Volatility & Squeeze — Bollinger Squeeze detection (BB inside Keltner = "
            "coiling for big move), ATR trend, historical volatility percentile, vol regime.\n"
            "(7) Signal Synthesis — Empirically-weighted confluence engine combining all 6 "
            "dimensions. Setup detection: mean_reversion_buy, trend_continuation, "
            "squeeze_breakout, distribution_warning, exhaustion_sell. Macro-conditioned "
            "conviction scoring. Outputs: action (strong_buy → strong_sell), position bias.\n\n"
            "IMPORTANT: After a refresh, the Director has cached TA signals for the full "
            "S&P 500 universe. For screening questions ('which stocks have squeezes?', "
            "'show me high conviction setups'), CHECK THE CACHED TA CONTEXT FIRST before "
            "calling this tool. Only call this tool for specific stocks not in the cache "
            "or when you need fresh computation.\n\n"
            "Use this for: 'What is the trend for AAPL?', 'Any squeeze breakouts forming?', "
            "'Which stocks have the strongest momentum?', 'Show me mean-reversion setups', "
            "'Is the rally in TSLA confirmed by volume?', 'What are the highest conviction "
            "signals?', 'Any distribution warnings in tech?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The technical analysis question",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific symbols to analyze. ALWAYS pass this for individual "
                        "stock analysis (e.g., ['AAPL', 'MSFT', 'NVDA']). If omitted, "
                        "defaults to 19 major ETFs for a broad market overview. All ~500 "
                        "S&P 500 constituents are available."
                    ),
                },
                "context": {
                    "type": "object",
                    "description": "Additional context such as timeframe preference",
                    "default": {},
                },
            },
            "required": ["question"],
        },
    }
