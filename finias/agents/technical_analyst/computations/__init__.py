"""
Technical Analyst computation modules.

Each module is a pure Python function that takes a pandas DataFrame
of OHLCV data and returns a typed dataclass with computed signals.

Modules are designed to run sequentially:
  trend → momentum (uses trend regime for adaptive thresholds) → levels (uses both)

All use the indicators.py module (pure pandas/numpy) for indicator computation.
"""
