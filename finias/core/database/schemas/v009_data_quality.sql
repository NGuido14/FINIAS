-- FINIAS Schema v009: Data Quality Tracking
-- Stores data quality report alongside each regime assessment
-- for historical audit: "was data quality degraded when this assessment was produced?"

ALTER TABLE regime_assessments ADD COLUMN IF NOT EXISTS data_quality_json JSONB;

-- Comment for documentation
COMMENT ON COLUMN regime_assessments.data_quality_json IS
    'DataQualityReport from validation layer. Tracks gaps, staleness, bounds violations at assessment time.';
