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
            "Query the Technical Analyst for price-based analysis. By default "
            "analyzes 19 major ETFs (SPY, QQQ, sector ETFs) for fast response. "
            "To analyze specific S&P 500 stocks, pass their tickers in the 'symbols' "
            "parameter (e.g., symbols=['AAPL', 'NVDA', 'TSLA']). Data is available "
            "for all ~500 S&P 500 constituents. All computation is pure Python using "
            "daily OHLCV data — results are precise and deterministic, not LLM-generated. "
            "\n\n"
            "CAPABILITIES:\n"
            "(1) Multi-Timeframe Trend Analysis — Ichimoku Cloud (tenkan/kijun cross, "
            "cloud color, price vs cloud, future cloud projection), ADX trend strength, "
            "SMA/EMA constellation (8/21/50/200), MA slope analysis, trend regime "
            "classification (strong_uptrend → strong_downtrend), trend maturity tracking, "
            "higher-highs/higher-lows structure detection.\n"
            "(2) Regime-Adaptive Momentum — RSI with adaptive overbought/oversold thresholds "
            "(shifts based on trend regime), MACD signal/histogram momentum, Stochastic "
            "%K/%D, Rate of Change at multiple timeframes, momentum divergence detection "
            "(regular bullish/bearish, hidden bullish/bearish), momentum thrust identification.\n"
            "(3) Support & Resistance — Classic and Fibonacci pivot points from OHLC, "
            "Bollinger Band dynamic levels, Donchian channel boundaries, Ichimoku cloud "
            "edges as forward S/R, key level clustering (multiple methods identifying same "
            "price zone), distance-to-nearest-support/resistance with risk/reward ratio.\n"
            "\n"
            "Use this for: 'What is the trend for AAPL?', 'Which stocks have the strongest "
            "momentum?', 'What are the key support/resistance levels for NVDA?', "
            "'Is the rally in TSLA technically confirmed?', 'Show me stocks in strong "
            "uptrends with bullish momentum', 'Any bearish divergences in tech stocks?'"
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
