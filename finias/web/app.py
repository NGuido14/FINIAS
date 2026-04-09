"""
FINIAS Web Interface — FastAPI Backend

Replaces the CLI with a browser-based interface.
Wraps the existing Director and Macro Strategist for chat and refresh.

Usage:
    python -m finias.web.app
    → Opens http://localhost:8000
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("finias.web")

# Global state — initialized on startup
_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize FINIAS on startup, cleanup on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Initializing FINIAS web interface...")

    from finias.core.config.settings import get_settings
    from finias.core.database.connection import DatabasePool
    from finias.core.database.migrations import run_migrations
    from finias.core.state.redis_state import RedisState
    from finias.core.agents.registry import ToolRegistry
    from finias.data.providers.polygon_client import PolygonClient
    from finias.data.providers.fred_client import FredClient
    from finias.data.cache.market_cache import MarketDataCache
    from finias.agents.macro_strategist.agent import MacroStrategist
    from finias.agents.technical_analyst.agent import TechnicalAnalyst
    from finias.agents.director.agent import Director

    db = DatabasePool()
    await db.initialize()
    await run_migrations(db)

    state = RedisState()
    await state.initialize()

    polygon = PolygonClient()
    fred = FredClient()
    cache = MarketDataCache(db=db, state=state, polygon=polygon, fred=fred)

    registry = ToolRegistry()
    macro = MacroStrategist(cache=cache, state=state)
    registry.register(macro)

    ta = TechnicalAnalyst(cache=cache, state=state)
    registry.register(ta)

    director = Director(registry=registry, state=state, db=db)

    _components.update({
        "db": db,
        "state": state,
        "polygon": polygon,
        "fred": fred,
        "cache": cache,
        "registry": registry,
        "director": director,
        "macro": macro,
    })

    logger.info("FINIAS web interface ready at http://localhost:8000")
    yield

    # Cleanup
    await polygon.close()
    await fred.close()
    await state.close()
    await db.close()
    logger.info("FINIAS web interface shut down.")


app = FastAPI(title="FINIAS", lifespan=lifespan)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>FINIAS</h1><p>static/index.html not found</p>")


@app.post("/api/chat")
async def chat(request: Request):
    """Send a message to the Director and get a response."""
    body = await request.json()
    message = body.get("message", "").strip()

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    director = _components.get("director")
    if not director:
        return JSONResponse({"error": "System not initialized"}, status_code=503)

    try:
        response = await director.chat(message)
        return JSONResponse({
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/tickers")
async def search_tickers(q: str = ""):
    """Search available tickers from the database."""
    db = _components.get("db")
    if not db:
        return JSONResponse({"tickers": []})

    try:
        if q:
            rows = await db.fetch(
                "SELECT DISTINCT symbol FROM market_data_daily WHERE UPPER(symbol) LIKE $1 ORDER BY symbol LIMIT 8",
                f"%{q.upper()}%"
            )
        else:
            rows = await db.fetch(
                "SELECT DISTINCT symbol FROM market_data_daily ORDER BY symbol LIMIT 8"
            )
        return JSONResponse({"tickers": [r["symbol"] for r in rows]})
    except Exception as e:
        logger.error(f"Ticker search error: {e}")
        return JSONResponse({"tickers": []})


@app.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get full price history and stats for a single stock."""
    db = _components.get("db")
    if not db:
        return JSONResponse({"error": "DB not initialized"}, status_code=503)

    try:
        symbol = symbol.upper()

        # Get all price history
        rows = await db.fetch(
            """SELECT trade_date, open, high, low, close, volume
            FROM market_data_daily
            WHERE symbol = $1
            ORDER BY trade_date ASC""",
            symbol
        )

        if not rows:
            return JSONResponse({"error": f"No data for {symbol}"}, status_code=404)

        prices = [{"date": str(r["trade_date"]), "open": float(r["open"]) if r["open"] else None,
                    "high": float(r["high"]) if r["high"] else None, "low": float(r["low"]) if r["low"] else None,
                    "close": float(r["close"]) if r["close"] else None,
                    "volume": int(r["volume"]) if r["volume"] else 0} for r in rows]

        latest = prices[-1]["close"] if prices else 0
        latest_date = prices[-1]["date"] if prices else ""
        latest_vol = prices[-1]["volume"] if prices else 0

        # Compute returns for various timeframes
        def pct_change(days_back):
            if len(prices) <= days_back:
                return None
            old = prices[-(days_back + 1)]["close"]
            if old and old > 0:
                return round((latest - old) / old * 100, 2)
            return None

        # High/low over last year
        year_prices = [p["close"] for p in prices[-252:] if p["close"]]
        high_52w = max(year_prices) if year_prices else None
        low_52w = min(year_prices) if year_prices else None

        # Average volume 20d
        recent_vols = [p["volume"] for p in prices[-20:] if p["volume"]]
        avg_vol_20d = int(sum(recent_vols) / len(recent_vols)) if recent_vols else 0

        return JSONResponse({
            "symbol": symbol,
            "latest_price": latest,
            "latest_date": latest_date,
            "volume": latest_vol,
            "avg_volume_20d": avg_vol_20d,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "returns": {
                "1d": pct_change(1),
                "5d": pct_change(5),
                "1m": pct_change(21),
                "3m": pct_change(63),
                "6m": pct_change(126),
                "1y": pct_change(252),
                "ytd": None,  # Would need Jan 1 lookup
            },
            "prices": prices,
        })
    except Exception as e:
        logger.error(f"Stock data error for {symbol}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/refresh")
async def refresh():
    """Run a full macro refresh."""
    macro = _components.get("macro")
    state = _components.get("state")
    db = _components.get("db")

    if not macro:
        return JSONResponse({"error": "System not initialized"}, status_code=503)

    try:
        # Fetch live prices
        price_status = "unavailable"
        try:
            from finias.data.providers.price_feed import fetch_live_prices, store_live_prices, backfill_from_live_prices
            lp = await fetch_live_prices()
            await store_live_prices(state, lp)
            fetched = sum(1 for k, v in lp.items() if k not in ("fetched_at", "source", "error") and v is not None)
            price_status = f"{fetched}/7 instruments"
            # Backfill FRED gaps with yfinance values
            backfill_result = await backfill_from_live_prices(db, state)
            bf_count = backfill_result.get("backfilled_count", 0)
            if bf_count > 0:
                price_status += f" (+{bf_count} backfilled)"
                logger.info(f"Backfilled {bf_count} FRED gaps from yfinance")
        except Exception as e:
            price_status = f"error: {e}"

        # Fetch COT data
        cot_status = "unavailable"
        try:
            from finias.data.providers.cot_client import fetch_and_store_cot_data
            cot_result = await fetch_and_store_cot_data(db)
            if cot_result.get("new_data"):
                cot_status = f"{cot_result['new_records']} new records"
            else:
                cot_status = f"up to date"
        except Exception as e:
            cot_status = f"error: {e}"

        # Run macro pipeline
        from finias.core.agents.models import AgentQuery
        from finias.agents.macro_strategist.prompts.refresh import MORNING_REFRESH_PROMPT

        query = AgentQuery(
            asking_agent="web_refresh",
            target_agent="macro_strategist",
            question=MORNING_REFRESH_PROMPT.format(question=""),
            require_fresh_data=True,
        )
        opinion = await macro.query(query)

        return JSONResponse({
            "status": "complete",
            "direction": opinion.direction.value,
            "confidence": opinion.confidence.value,
            "regime": opinion.regime.value if opinion.regime else None,
            "findings": len(opinion.key_findings),
            "prices": price_status,
            "cot": cot_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/status")
async def status():
    """Get current system status from Redis."""
    state = _components.get("state")
    if not state:
        return JSONResponse({"error": "System not initialized"}, status_code=503)

    try:
        regime = await state.get_regime()
        if not regime:
            return JSONResponse({"status": "no_data", "message": "No regime cached. Run a refresh."})

        # Extract key summary fields
        traj = regime.get("trajectory", {})
        bias = traj.get("forward_bias", {})
        sizing = traj.get("position_sizing", {})
        interp = regime.get("interpretation", {})
        kl = regime.get("key_levels", {})
        reg = regime.get("regime", {})
        pos = regime.get("positioning", {}).get("aggregate", {})

        return JSONResponse({
            "status": "live",
            "updated_at": regime.get("_updated_at"),
            "regime": reg.get("primary", "unknown") if isinstance(reg, dict) else str(reg),
            "composite": kl.get("composite_score", 0),
            "forward_bias": bias.get("bias", "neutral"),
            "confidence": bias.get("confidence", "low"),
            "vix": kl.get("vix"),
            "recession_prob": kl.get("recession_prob"),
            "beta_target": sizing.get("portfolio_beta_target"),
            "max_position": sizing.get("max_single_position_pct"),
            "reduce_exposure": sizing.get("reduce_overall_exposure"),
            "findings_count": len(interp.get("key_findings", [])),
            "risks_count": len(interp.get("risks", [])),
            "positioning_score": pos.get("score"),
            "crowding_alerts": pos.get("crowding_alert_count"),
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/reset")
async def reset_conversation():
    """Reset the Director's conversation history."""
    director = _components.get("director")
    if director:
        director.reset_conversation()
    return JSONResponse({"status": "reset"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "finias.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
