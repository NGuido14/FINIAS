-- FINIAS Schema v004: Remove unused tables
-- These tables were created in v001/v002 but the agent never writes to them.
-- The data they were designed for already exists in agent_opinions (JSONB)
-- and the macro_data_matrix (pivoted view).

-- regime_history: superseded by regime_assessments, neither is populated
DROP TABLE IF EXISTS regime_history;

-- Domain-specific history tables: never populated by agent code.
-- All this data is available in agent_opinions.data_points JSONB.
DROP TABLE IF EXISTS cycle_phase_history;
DROP TABLE IF EXISTS monetary_policy_history;
DROP TABLE IF EXISTS inflation_history;

-- Keep regime_assessments — we'll wire the agent to write to it.
-- Keep agent_opinions — audit trail.
-- Keep agent_health_log — health monitoring.
