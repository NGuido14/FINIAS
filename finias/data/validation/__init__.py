"""
FINIAS Data Validation Layer — Shared Infrastructure.

Provides data quality checks for all agents. Catches silent data corruption
(missing observations, gaps in time series, stale data, outliers) before
it reaches computation modules.

Core principle: warn, never block. The pipeline always runs.
Quality issues flow to Claude via data notes and to downstream agents
via the existing data_freshness_warnings field on MacroContext.

Usage:
    from finias.data.validation.quality import check_series_gaps, DataQualityReport
    from finias.data.validation.fred_quality import validate_all_fred
    from finias.data.validation.polygon_quality import validate_price_bars
    from finias.data.validation.bounds import check_computation_bounds
"""
