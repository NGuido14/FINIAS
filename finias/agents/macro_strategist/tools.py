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
            "Query the Macro Strategist for comprehensive Phase 1 macro-economic analysis across 8 domains. "
            "This agent performs multi-dimensional regime assessment covering: business cycle (LEI, ISM, unemployment), "
            "monetary policy (Fed funds, balance sheet), liquidity conditions (net Fed liquidity, reverse repo), "
            "inflation (CPI/PCE trends, expectations), yield curve (term structure, spreads), credit markets (HY spreads, NFCI), "
            "labor markets (jobless claims, quits, participation), and cross-asset signals (VIX, breadth, sector rotation). "
            "It identifies the binding constraint limiting market performance or policy optionality. "
            "Ask it about: current macro regime, recession risk, Fed policy trajectory, inflation outlook, "
            "liquidity conditions, sector allocation implications, or how macro conditions affect specific trades."
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
