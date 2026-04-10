"""Show current TA signal distribution after synthesis recalibration."""
import asyncio
import json

from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.core.state.redis_state import RedisState
from finias.data.cache.market_cache import MarketDataCache
from finias.data.providers.polygon_client import PolygonClient
from finias.data.providers.fred_client import FredClient
from finias.agents.technical_analyst.agent import TechnicalAnalyst
from finias.core.agents.models import AgentQuery


async def main():
    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)
    state = RedisState()
    await state.initialize()
    polygon = PolygonClient()
    fred = FredClient()
    cache = MarketDataCache(db, state, polygon, fred)

    ta = TechnicalAnalyst(cache=cache, state=state)

    from finias.data.universe import get_active_symbols
    symbols = await get_active_symbols(db)
    print("Running TA on {} symbols...".format(len(symbols)))

    query = AgentQuery(
        asking_agent="script",
        target_agent="technical_analyst",
        question="full refresh",
        context={"symbols": symbols},
    )
    opinion = await ta.query(query)

    raw = await state.client.get("ta:current")
    data = json.loads(raw)
    signals = data.get("signals", {})

    actions = {}
    setups = {}
    for sym, sig in signals.items():
        synth = sig.get("synthesis", {})
        a = synth.get("action", "unknown")
        s = synth.get("setup", {}).get("type", "unknown")
        actions[a] = actions.get(a, 0) + 1
        setups[s] = setups.get(s, 0) + 1

    print("\nTotal: {}".format(len(signals)))
    print("\nACTIONS:")
    for a, n in sorted(actions.items(), key=lambda x: -x[1]):
        print("  {}: {} ({:.1f}%)".format(a, n, n / len(signals) * 100))
    print("\nSETUPS:")
    for s, n in sorted(setups.items(), key=lambda x: -x[1]):
        print("  {}: {} ({:.1f}%)".format(s, n, n / len(signals) * 100))

    print("\n" + "=" * 60)
    print("STRONG BUY:")
    for sym, sig in sorted(signals.items()):
        synth = sig.get("synthesis", {})
        if synth.get("action") == "strong_buy":
            setup = synth.get("setup", {}).get("type", "?")
            conv = synth.get("conviction", {}).get("score", 0)
            macro = synth.get("macro", {}).get("alignment", "?")
            trend = sig.get("trend", {}).get("trend_regime", "?")
            rsi = sig.get("momentum", {}).get("rsi", {}).get("value", "?")
            print("  {}: {}, conv={:.2f}, macro={}, trend={}, RSI={}".format(
                sym, setup, conv, macro, trend, rsi))

    print("\nBUY:")
    for sym, sig in sorted(signals.items()):
        synth = sig.get("synthesis", {})
        if synth.get("action") == "buy":
            setup = synth.get("setup", {}).get("type", "?")
            conv = synth.get("conviction", {}).get("score", 0)
            trend = sig.get("trend", {}).get("trend_regime", "?")
            rsi = sig.get("momentum", {}).get("rsi", {}).get("value", "?")
            print("  {}: {}, conv={:.2f}, trend={}, RSI={}".format(
                sym, setup, conv, trend, rsi))

    print("\nSELL/REDUCE:")
    for sym, sig in sorted(signals.items()):
        synth = sig.get("synthesis", {})
        if synth.get("action") in ("sell", "strong_sell", "reduce"):
            setup = synth.get("setup", {}).get("type", "?")
            conv = synth.get("conviction", {}).get("score", 0)
            trend = sig.get("trend", {}).get("trend_regime", "?")
            print("  {}: {}, {}, conv={:.2f}, trend={}".format(
                sym, synth["action"], setup, conv, trend))

    await fred.close()
    await polygon.close()
    await state.close()
    await db.close()


asyncio.run(main())