-- FINIAS Schema v007: Backtest Trajectory Extension
-- Adds trajectory signal storage to backtest results for out-of-sample validation

ALTER TABLE backtest_results ADD COLUMN IF NOT EXISTS trajectory_json JSONB;
