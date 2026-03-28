# FINIAS — Financial Intelligence Agency System

An agentic AI system where specialized agents work together like a world-class trading firm.

## Architecture

```
Director (Layer 3) — User-facing conversational interface
    ↓ routes queries via Claude tool_use
Macro Strategist (Layer 1) — Domain expert
    ↓ fetches data, runs computations
Python Computations (Layer 0) — Pure math
    ↓ yield curve, volatility, breadth, cross-asset
Data Providers — Polygon.io, FRED
```

## Quick Start

1. Copy `.env.example` to `.env` and fill in your API keys
2. Start PostgreSQL and Redis
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python -m finias.interface.cli`

## Sprint 0: The Vertical Slice

Ask "How is the market today?" and receive a grounded, data-driven answer assembled by an AI Director that consulted a Macro Strategist agent backed by real market data.

## Tech Stack

- Python 3.11+
- PostgreSQL (durable storage)
- Redis (ephemeral state)
- Anthropic Claude API
- Polygon.io (market data)
- FRED (economic data)
