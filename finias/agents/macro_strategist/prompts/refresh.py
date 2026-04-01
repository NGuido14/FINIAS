"""
Morning refresh prompt for the Macro Strategist.

Used by the scheduled daily refresh (6:30 AM) to produce a comprehensive
macro assessment that gets cached and served to all subsequent queries
and downstream agents throughout the day.

This prompt is NOT used for interactive CLI queries — those pass the
user's actual question. This is the standardized comprehensive assessment.
"""

MORNING_REFRESH_PROMPT = """Provide a comprehensive morning macro assessment covering ALL of the following domains systematically. Do not skip any section.

1. REGIME STATUS: Current regime classification, composite score, confidence level. Has anything changed materially from recent conditions? What is the primary regime risk — transition toward risk-off, or stability?

2. BINDING CONSTRAINT: What is the single most important factor limiting market performance right now? How close are we to a shift in the binding constraint? What would cause it to change?

3. INFLATION ASSESSMENT: Core PCE level and trajectory (YoY vs 3-month annualized). Is inflation accelerating or decelerating? How far from the Fed's target? Is the inflation surprise hawkish or dovish? What does the sticky vs flexible decomposition tell us?

4. MONETARY POLICY & LIQUIDITY: Fed funds rate, balance sheet direction (growing/shrinking), net liquidity level and trend, NFCI conditions. Is policy stance restrictive, neutral, or accommodative? Is liquidity supporting or constraining risk assets?

5. GROWTH & CYCLE: Business cycle phase, leading indicators, labor market signals. What does the Sahm Rule distance tell us about recession risk? Is the custom leading indicator confirming or diverging from the cycle phase?

6. VOLATILITY & MARKET STRUCTURE: VIX level and velocity, term structure (contango vs backwardation), variance risk premium, sector correlation regime. Is volatility stress persistent or transient?

7. CROSS-ASSET SIGNALS: Walk through the key cross-asset relationships — Dollar-Equity, Credit-Equity, Oil-Equity, Gold-Equity, stock-bond correlation. Which pairs are confirming the regime assessment? Which are diverging? Use the EXACT computed correlation values and betas.

8. POSITION GUIDANCE: Current position sizing limits (max position %, beta target, cash target). Are we in reduce_exposure mode? What is the pre-event sizing multiplier and what events are upcoming?

9. RISK HIERARCHY: Rank the scenario triggers by proximity and consequence. Which trigger is most likely to fire in the next 30 days? What would happen to the portfolio if it fires?

10. FORWARD OUTLOOK: Using the trajectory signals (forward bias, inflation trajectory, stress contrarian), what is the directional lean and confidence? What specific conditions would shift the outlook?

11. WEB SEARCH VERIFICATION: Search for current values of any data that may be materially stale given market conditions. Specifically verify: current VIX, oil prices (both WTI AND Brent — note the spread), credit spreads, and any breaking geopolitical developments affecting markets. Also check Atlanta Fed GDPNow for the current quarter estimate.

Be specific. Cite exact numbers. Explain cross-domain connections. Identify conflicts between signals. Think like a macro strategist at a top hedge fund.

{question}"""
