-- FINIAS Schema v006: Full Output Persistence
-- Store complete regime JSON and Claude interpretation for downstream agents

ALTER TABLE regime_assessments ADD COLUMN IF NOT EXISTS full_regime_json JSONB;
ALTER TABLE regime_assessments ADD COLUMN IF NOT EXISTS interpretation_json JSONB;
