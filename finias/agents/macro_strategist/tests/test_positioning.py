"""Tests for CFTC positioning computation module."""

import pytest
from datetime import date, timedelta

from finias.agents.macro_strategist.computations.positioning import (
    compute_contract_positioning,
    compute_positioning_analysis,
    generate_positioning_data_notes,
    ContractPositioning,
    PositioningAnalysis,
    CONTRACT_WEIGHTS,
)


# ============================================================================
# Helper: Generate synthetic COT history
# ============================================================================

def _make_history(net_specs: list[int], start_date: date = None) -> list[dict]:
    """Create synthetic COT history from a list of net spec values."""
    if start_date is None:
        start_date = date(2023, 1, 3)
    return [
        {
            "report_date": start_date + timedelta(weeks=i),
            "net_spec": ns,
            "open_interest": abs(ns) * 10,
        }
        for i, ns in enumerate(net_specs)
    ]


# ============================================================================
# 1. Contract Positioning Tests
# ============================================================================

class TestContractPositioning:
    def test_empty_history(self):
        """No data → defaults."""
        cp = compute_contract_positioning("sp500", [])
        assert cp.contract_key == "sp500"
        assert cp.net_spec == 0
        assert cp.net_spec_percentile == 50.0
        assert cp.crowding == "neutral"
        assert cp.confidence == "low"

    def test_percentile_at_maximum(self):
        """Current value is the highest → 100th percentile."""
        history = _make_history([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cp = compute_contract_positioning("sp500", history)
        assert cp.net_spec_percentile == 100.0
        assert cp.net_spec == 100

    def test_percentile_at_minimum(self):
        """Current value is the lowest → near 0th percentile."""
        history = _make_history([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        cp = compute_contract_positioning("sp500", history)
        assert cp.net_spec_percentile == 10.0  # 1/10 = 10%
        assert cp.net_spec == 10

    def test_crowded_long(self):
        """Percentile > 90 → crowded_long."""
        # Make current value the highest in 100 observations
        specs = list(range(-50, 50))  # -50 to 49, 100 values
        specs.append(100)  # 101st value, clearly the highest
        history = _make_history(specs)
        cp = compute_contract_positioning("sp500", history)
        assert cp.crowding == "crowded_long"
        assert cp.net_spec_percentile > 90

    def test_crowded_short(self):
        """Percentile < 10 → crowded_short."""
        specs = list(range(-50, 50))
        specs.append(-100)  # Clearly the lowest
        history = _make_history(specs)
        cp = compute_contract_positioning("sp500", history)
        assert cp.crowding == "crowded_short"
        assert cp.net_spec_percentile < 10

    def test_neutral_positioning(self):
        """Middle of range → neutral."""
        specs = list(range(0, 100))
        specs.append(50)  # Middle value
        history = _make_history(specs)
        cp = compute_contract_positioning("sp500", history)
        assert cp.crowding == "neutral"
        assert 10 < cp.net_spec_percentile < 90

    def test_rate_of_change_positive(self):
        """Increasing net spec → positive rate of change."""
        history = _make_history([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cp = compute_contract_positioning("sp500", history)
        assert cp.rate_of_change_4w > 0  # 100 - 60 = 40

    def test_rate_of_change_negative(self):
        """Decreasing net spec → negative rate of change."""
        history = _make_history([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        cp = compute_contract_positioning("sp500", history)
        assert cp.rate_of_change_4w < 0  # 10 - 50 = -40

    def test_confidence_high(self):
        """156+ weeks → high confidence."""
        history = _make_history([i for i in range(160)])
        cp = compute_contract_positioning("sp500", history)
        assert cp.confidence == "high"
        assert cp.lookback_weeks == 160

    def test_confidence_moderate(self):
        """52-155 weeks → moderate confidence."""
        history = _make_history([i for i in range(100)])
        cp = compute_contract_positioning("sp500", history)
        assert cp.confidence == "moderate"

    def test_confidence_low(self):
        """<52 weeks → low confidence."""
        history = _make_history([i for i in range(30)])
        cp = compute_contract_positioning("sp500", history)
        assert cp.confidence == "low"

    def test_to_dict_structure(self):
        """to_dict returns all expected keys."""
        history = _make_history([10, 20, 30, 40, 50])
        cp = compute_contract_positioning("sp500", history)
        d = cp.to_dict()
        assert "contract_key" in d
        assert "net_spec" in d
        assert "net_spec_percentile" in d
        assert "crowding" in d
        assert "rate_of_change_4w" in d
        assert "lookback_weeks" in d
        assert "confidence" in d
        assert "_note" in d


# ============================================================================
# 2. Aggregate Positioning Tests
# ============================================================================

class TestPositioningAnalysis:
    def test_empty_data(self):
        """No contract data → default neutral."""
        result = compute_positioning_analysis({}, staleness_days=999)
        assert result.aggregate_score == 0.0
        assert result.sp500_positioning_signal == "neutral"
        assert result.crowding_alert_count == 0

    def test_aggregate_score_net_short(self):
        """All contracts at low percentiles → negative aggregate."""
        data = {}
        for key in CONTRACT_WEIGHTS:
            # All at ~5th percentile (very short)
            specs = list(range(50, 150))  # 50-149
            specs.append(10)  # Current value is lowest
            data[key] = _make_history(specs)
        result = compute_positioning_analysis(data, staleness_days=5)
        assert result.aggregate_score < -0.5
        assert result.crowding_alert_count >= 3

    def test_aggregate_score_net_long(self):
        """All contracts at high percentiles → positive aggregate."""
        data = {}
        for key in CONTRACT_WEIGHTS:
            specs = list(range(0, 100))
            specs.append(200)  # Current value is highest
            data[key] = _make_history(specs)
        result = compute_positioning_analysis(data, staleness_days=5)
        assert result.aggregate_score > 0.5

    def test_sp500_signal_constructive(self):
        """S&P 500 at <10th percentile → constructive."""
        specs = list(range(0, 200))
        specs.append(-100)  # Way below everything
        data = {"sp500": _make_history(specs)}
        result = compute_positioning_analysis(data, staleness_days=5)
        assert result.sp500_positioning_signal == "constructive"

    def test_sp500_signal_cautious(self):
        """S&P 500 at >90th percentile → cautious."""
        specs = list(range(0, 200))
        specs.append(500)  # Way above everything
        data = {"sp500": _make_history(specs)}
        result = compute_positioning_analysis(data, staleness_days=5)
        assert result.sp500_positioning_signal == "cautious"

    def test_sp500_signal_neutral(self):
        """S&P 500 in middle → neutral."""
        specs = list(range(0, 200))
        specs.append(100)  # Middle
        data = {"sp500": _make_history(specs)}
        result = compute_positioning_analysis(data, staleness_days=5)
        assert result.sp500_positioning_signal == "neutral"

    def test_sp500_signal_low_confidence_neutral(self):
        """S&P 500 with low confidence (< 52 weeks) → always neutral."""
        specs = [10, 20, 30]  # Only 3 weeks
        specs.append(-100)
        data = {"sp500": _make_history(specs)}
        result = compute_positioning_analysis(data, staleness_days=5)
        # Low confidence should force neutral regardless of percentile
        assert result.sp500_positioning_signal == "neutral"

    def test_contract_weights_sum_to_one(self):
        """Verify contract weights sum to 1.0."""
        assert abs(sum(CONTRACT_WEIGHTS.values()) - 1.0) < 0.001

    def test_to_dict_structure(self):
        """to_dict returns nested structure."""
        data = {"sp500": _make_history([10, 20, 30, 40, 50])}
        result = compute_positioning_analysis(data, staleness_days=5)
        d = result.to_dict()
        assert "contracts" in d
        assert "aggregate" in d
        assert "score" in d["aggregate"]
        assert "sp500_positioning_signal" in d["aggregate"]


# ============================================================================
# 3. Divergence Detection Tests
# ============================================================================

class TestDivergences:
    def test_risk_on_safe_haven_contradiction(self):
        """Crowded long equities AND gold → divergence."""
        specs_high = list(range(0, 200))
        specs_high.append(500)  # Crowded long
        data = {
            "sp500": _make_history(specs_high),
            "gold": _make_history(specs_high),
        }
        result = compute_positioning_analysis(data, staleness_days=5)
        assert len(result.divergences) >= 1
        assert any("equities AND gold" in d for d in result.divergences)

    def test_no_divergence_normal(self):
        """Normal positioning across assets → no divergences."""
        specs_mid = list(range(0, 200))
        specs_mid.append(100)  # Middle
        data = {
            "sp500": _make_history(specs_mid),
            "gold": _make_history(specs_mid),
        }
        result = compute_positioning_analysis(data, staleness_days=5)
        assert len(result.divergences) == 0


# ============================================================================
# 4. Data Notes Tests
# ============================================================================

class TestDataNotes:
    def test_no_data_note(self):
        """Empty positioning → 'No data available' note."""
        result = PositioningAnalysis()
        notes = generate_positioning_data_notes(result)
        assert len(notes) == 1
        assert "No CFTC COT data available" in notes[0]

    def test_notes_contain_all_contracts(self):
        """Notes should mention all provided contracts."""
        data = {}
        for key in CONTRACT_WEIGHTS:
            data[key] = _make_history(list(range(100)))
        result = compute_positioning_analysis(data, staleness_days=5)
        notes = generate_positioning_data_notes(result)
        full_text = "\n".join(notes)
        assert "S&P 500" in full_text
        assert "10Y Treasury" in full_text
        assert "WTI Crude" in full_text
        assert "Gold" in full_text
        assert "Dollar Index" in full_text

    def test_notes_flag_crowding(self):
        """Crowded contracts get flagged in notes."""
        specs = list(range(0, 200))
        specs.append(500)
        data = {"sp500": _make_history(specs)}
        result = compute_positioning_analysis(data, staleness_days=5)
        notes = generate_positioning_data_notes(result)
        full_text = "\n".join(notes)
        assert "CROWDED LONG" in full_text

    def test_notes_staleness_label(self):
        """Staleness days produce correct label."""
        data = {"sp500": _make_history(list(range(100)))}
        # Normal staleness
        result = compute_positioning_analysis(data, staleness_days=5)
        notes = generate_positioning_data_notes(result)
        assert "normal" in "\n".join(notes)
        # Stale
        result2 = compute_positioning_analysis(data, staleness_days=15)
        notes2 = generate_positioning_data_notes(result2)
        assert "stale" in "\n".join(notes2).lower()

    def test_notes_contrarian_context(self):
        """Crowded short S&P 500 → contrarian note."""
        specs = list(range(50, 250))
        specs.append(-100)
        data = {"sp500": _make_history(specs)}
        result = compute_positioning_analysis(data, staleness_days=5)
        notes = generate_positioning_data_notes(result)
        full_text = "\n".join(notes)
        assert "CONTRARIAN" in full_text
        assert "asymmetry to upside" in full_text


# ============================================================================
# 5. COT Client Tests (import verification)
# ============================================================================

class TestCOTClient:
    def test_cot_client_imports(self):
        """Verify the COT client can be imported."""
        from finias.data.providers.cot_client import (
            COT_CONTRACTS,
            fetch_and_store_cot_data,
            get_cot_history,
            get_latest_cot,
            get_cot_staleness_days,
        )
        assert len(COT_CONTRACTS) == 5
        assert "sp500" in COT_CONTRACTS
        assert "treasury_10y" in COT_CONTRACTS
        assert "wti_crude" in COT_CONTRACTS
        assert "gold" in COT_CONTRACTS
        assert "dollar_index" in COT_CONTRACTS

    def test_contract_names_verified(self):
        """Contract names must match what was verified against live CFTC data."""
        from finias.data.providers.cot_client import COT_CONTRACTS
        assert COT_CONTRACTS["sp500"] == "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE"
        assert COT_CONTRACTS["treasury_10y"] == "UST 10Y NOTE - CHICAGO BOARD OF TRADE"
        assert COT_CONTRACTS["wti_crude"] == "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE"
        assert COT_CONTRACTS["gold"] == "GOLD - COMMODITY EXCHANGE INC."
        assert COT_CONTRACTS["dollar_index"] == "USD INDEX - ICE FUTURES U.S."

    def test_cftc_cot_library_installed(self):
        """Verify the cftc-cot library is importable."""
        from cftc_cot import cot_download_year_range
        assert callable(cot_download_year_range)
