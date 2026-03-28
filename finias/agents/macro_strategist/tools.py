"""
Macro Strategist tool definitions.

This module exposes the Macro Strategist as a tool that higher-layer
agents (Director, Trade Decision Agent) can call through Claude's
tool_use mechanism.
"""

from finias.agents.macro_strategist.agent import MacroStrategist


def get_macro_tool_definition() -> dict:
    """
    Get the Claude tool_use definition for the Macro Strategist.

    This is called by the ToolRegistry when registering the agent.
    The definition tells Claude what the agent can do and what inputs it accepts.
    """
    return {
        "name": "query_macro_strategist",
        "description": (
            "Query the Macro Strategist for comprehensive macro-economic analysis across 8 domains "
            "using 68 FRED economic series and 19 Polygon market data symbols. "
            "Domains: "
            "(1) Yield Curve — spreads, real yields (TIPS), term premium, forward rate expectations "
            "(market-implied future rates vs current fed funds), recession signal scoring. "
            "(2) Volatility — VIX level/percentile, VIX term structure (contango vs backwardation via VIX3M), "
            "realized vs implied vol, variance risk premium (VRP), sector correlation regime. "
            "(3) Monetary Policy — Fed funds vs neutral rate, balance sheet trajectory, net liquidity "
            "(Fed assets minus TGA minus reverse repo), NFCI financial conditions, credit creation, policy stance. "
            "(4) Business Cycle — Sahm Rule recession indicator, custom leading indicator proxy "
            "(claims, permits, sentiment, hours), GDPNow real-time GDP estimate, cycle phase classification. "
            "(5) Inflation — Core PCE/CPI year-over-year and 3-month annualized (Fed's preferred measure), "
            "sticky vs flexible CPI decomposition, wage pressure, breakeven inflation expectations, "
            "wage-price spiral risk detection. "
            "(6) Market Breadth — Sector participation (% above 200/50 day MA across 11 sector ETFs), "
            "SPY/RSP cap-weighted vs equal-weight divergence, sector relative strength rankings, "
            "cyclical vs defensive rotation signals, sector dispersion. "
            "(7) Cross-Asset — US dollar trend, HY credit spreads, copper/gold ratio (growth proxy), "
            "oil dynamics (supply shock vs demand), stock-bond correlation (risk parity stress detection), "
            "IWM/SPY small cap vs large cap risk appetite, credit-equity divergence, EM stress. "
            "(8) Regime Detection — Hierarchical 4-category model (growth, monetary, inflation, market signals) "
            "with dynamic weighting, composite scoring, stress index, and binding constraint identification. "
            "Ask about: market regime, recession risk, yield curve signals, Fed policy outlook, inflation trajectory, "
            "VIX term structure, forward rate expectations, liquidity conditions, breadth health, sector rotation, "
            "cross-asset signals, risk parity stress, or how macro conditions affect specific sectors or trades."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The question to ask the Macro Strategist. Examples: "
                        "'What is the current market regime?', "
                        "'Is the yield curve signaling recession?', "
                        "'How does the macro environment look for tech stocks?'"
                    )
                },
                "context": {
                    "type": "object",
                    "description": "Additional context such as specific sectors or tickers of interest",
                    "default": {}
                },
                "require_fresh_data": {
                    "type": "boolean",
                    "description": "Force fresh data fetch instead of using cached data",
                    "default": False
                }
            },
            "required": ["question"]
        }
    }
