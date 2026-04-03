"""
FINIAS CLI Interface

Terminal-based conversational interface to the Director.
This is the Sprint 0 user interface — simple but functional.
"""

import asyncio
import logging
import sys

from finias.core.config.settings import get_settings
from finias.core.database.connection import DatabasePool
from finias.core.database.migrations import run_migrations
from finias.core.state.redis_state import RedisState
from finias.core.agents.registry import ToolRegistry
from finias.data.providers.polygon_client import PolygonClient
from finias.data.providers.fred_client import FredClient
from finias.data.cache.market_cache import MarketDataCache
from finias.agents.macro_strategist.agent import MacroStrategist
from finias.agents.director.agent import Director
from finias.core.agents.models import AgentQuery


BANNER = """
╔══════════════════════════════════════════════════════════╗
║                       F I N I A S                        ║
║          Financial Intelligence Agency System            ║
║                                                          ║
║  An agentic AI system that thinks, monitors, discovers.  ║
╚══════════════════════════════════════════════════════════╝

Type your questions in natural language.
Type 'quit' or 'exit' to end the session.
Type 'status' to check system health.
Type 'reset' to clear conversation history.
Type 'refresh' to run a full macro refresh (caches for fast queries).
"""


async def initialize_system():
    """Boot up all FINIAS components."""
    print("Initializing FINIAS...")

    # Database
    print("  → Connecting to PostgreSQL...")
    db = DatabasePool()
    await db.initialize()

    print("  → Running migrations...")
    await run_migrations(db)

    # Redis
    print("  → Connecting to Redis...")
    state = RedisState()
    await state.initialize()

    # Data providers
    print("  → Initializing data providers...")
    polygon = PolygonClient()
    fred = FredClient()
    cache = MarketDataCache(db=db, state=state, polygon=polygon, fred=fred)

    # Agents
    print("  → Initializing agents...")
    registry = ToolRegistry()

    macro = MacroStrategist(cache=cache, state=state)
    registry.register(macro)

    director = Director(registry=registry, state=state)

    print("  → System ready.\n")

    return {
        "db": db,
        "state": state,
        "polygon": polygon,
        "fred": fred,
        "cache": cache,
        "registry": registry,
        "director": director,
    }


async def shutdown_system(components: dict):
    """Cleanly shut down all components."""
    print("\nShutting down FINIAS...")

    if "polygon" in components:
        await components["polygon"].close()
    if "fred" in components:
        await components["fred"].close()
    if "state" in components:
        await components["state"].close()
    if "db" in components:
        await components["db"].close()

    print("Goodbye.")


async def main():
    """Main CLI loop."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("finias.log"),
            logging.StreamHandler(sys.stderr) if get_settings().log_level == "DEBUG" else logging.NullHandler(),
        ]
    )

    # Initialize
    try:
        components = await initialize_system()
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("Check your .env file and ensure PostgreSQL and Redis are running.")
        return

    director: Director = components["director"]
    registry: ToolRegistry = components["registry"]

    print(BANNER)

    # Show available agents
    agents = registry.list_agents()
    print(f"Online agents: {len(agents)}")
    for agent in agents:
        print(f"  • {agent['name']} ({agent['layer']}): {agent['description'][:60]}...")
    print()

    # Main loop
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                break

            if user_input.lower() == "status":
                print("\nSystem Status:")
                for agent_info in registry.list_agents():
                    agent = registry.get_agent(agent_info["name"])
                    if agent:
                        health = await agent.health_check()
                        status = "✓ healthy" if health.is_healthy else f"✗ {health.error_message}"
                        print(f"  {agent_info['name']}: {status}")
                print()
                continue

            if user_input.lower() == "reset":
                director.reset_conversation()
                print("Conversation history cleared.\n")
                continue

            if user_input.lower() == "refresh":
                print("\nRunning full macro refresh...")
                try:
                    macro = components["registry"].get_agent("macro_strategist")
                    if macro:
                        # Fetch live prices first
                        try:
                            from finias.data.providers.price_feed import fetch_live_prices, store_live_prices
                            lp = await fetch_live_prices()
                            await store_live_prices(components["state"], lp)
                            fetched_count = sum(1 for k, v in lp.items() if k not in ("fetched_at", "source", "error") and v is not None)
                            print(f"  Live prices fetched: {fetched_count}/7 instruments")
                        except Exception as e:
                            print(f"  Live prices unavailable: {e}")
                        from finias.agents.macro_strategist.prompts.refresh import MORNING_REFRESH_PROMPT
                        refresh_query = AgentQuery(
                            asking_agent="cli_refresh",
                            target_agent="macro_strategist",
                            question=MORNING_REFRESH_PROMPT.format(question=""),
                            require_fresh_data=True,
                        )
                        opinion = await macro.timed_query(refresh_query)
                        print(f"Refresh complete. Regime: {opinion.regime.value if opinion.regime else 'N/A'}, "
                              f"Confidence: {opinion.confidence.value}, "
                              f"Findings: {len(opinion.key_findings)}")
                        print("Cached context now available for fast queries.\n")
                    else:
                        print("Macro agent not available.\n")
                except Exception as e:
                    print(f"Refresh failed: {e}\n")
                continue

            # Process through Director
            print("\nFINIAS: ", end="", flush=True)
            try:
                response = await director.chat(user_input)
                print(response)
            except Exception as e:
                print(f"Error processing query: {e}")
                logging.exception("Error in director.chat")
            print()

    finally:
        await shutdown_system(components)


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
