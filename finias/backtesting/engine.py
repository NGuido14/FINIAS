"""
Walk-Forward Backtesting Engine

Steps through historical dates, enforces look-ahead prevention,
and delegates computation to agent-specific runners.

This engine is agent-agnostic. The runner (e.g., macro_runner) knows
which computations to perform. The engine handles the time loop,
data windowing, and result storage.
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Any, Callable, Awaitable
import json
import uuid
import logging

from finias.core.database.connection import DatabasePool

logger = logging.getLogger("finias.backtesting.engine")


async def run_walk_forward(
    db: DatabasePool,
    runner_fn: Callable[[date], Awaitable[dict[str, Any]]],
    start_date: date,
    end_date: date,
    step_days: int = 7,
    warmup_weeks: int = 52,
    forward_return_fn: Callable[[date], Awaitable[dict[str, float]]] = None,
) -> str:
    """
    Execute a walk-forward backtest.

    Args:
        db: Database pool for storing results
        runner_fn: Async function that takes a sim_date and returns regime scores dict
        start_date: First simulation date
        end_date: Last simulation date
        step_days: Days between each simulation (7 = weekly)
        warmup_weeks: Number of initial weeks to skip in scoring (builds history buffers)
        forward_return_fn: Async function that takes a sim_date and returns forward returns

    Returns:
        backtest_run_id (UUID string) for querying results
    """
    run_id = str(uuid.uuid4())
    warmup_end = start_date + timedelta(weeks=warmup_weeks)
    total_steps = 0
    scored_steps = 0

    logger.info(f"Starting backtest run {run_id[:8]}...")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Step: {step_days} days")
    logger.info(f"  Warmup: {warmup_weeks} weeks (until {warmup_end})")

    current = start_date
    while current <= end_date:
        is_warmup = current < warmup_end

        try:
            # Run the computation pipeline for this date
            scores = await runner_fn(current)

            if scores is None:
                logger.warning(f"Runner returned None for {current}, skipping")
                current += timedelta(days=step_days)
                continue

            # Get forward returns (only if we have future data)
            fwd_returns = {}
            if forward_return_fn and not is_warmup:
                fwd_returns = await forward_return_fn(current)

            # Store results
            await db.execute(
                """
                INSERT INTO backtest_results (
                    backtest_run_id, sim_date,
                    composite_score, growth_score, monetary_score,
                    inflation_score, market_score,
                    primary_regime, cycle_phase, stress_index,
                    confidence, binding_constraint,
                    spx_fwd_5d, spx_fwd_20d, spx_fwd_60d, spx_max_dd_20d,
                    modules_used, warmup, trajectory_json
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19
                )
                ON CONFLICT (backtest_run_id, sim_date) DO UPDATE SET
                    composite_score = EXCLUDED.composite_score,
                    growth_score = EXCLUDED.growth_score,
                    monetary_score = EXCLUDED.monetary_score,
                    inflation_score = EXCLUDED.inflation_score,
                    market_score = EXCLUDED.market_score,
                    primary_regime = EXCLUDED.primary_regime,
                    cycle_phase = EXCLUDED.cycle_phase,
                    stress_index = EXCLUDED.stress_index,
                    confidence = EXCLUDED.confidence,
                    binding_constraint = EXCLUDED.binding_constraint,
                    spx_fwd_5d = EXCLUDED.spx_fwd_5d,
                    spx_fwd_20d = EXCLUDED.spx_fwd_20d,
                    spx_fwd_60d = EXCLUDED.spx_fwd_60d,
                    spx_max_dd_20d = EXCLUDED.spx_max_dd_20d,
                    trajectory_json = EXCLUDED.trajectory_json
                """,
                run_id, current,
                scores.get("composite_score"),
                scores.get("growth_score"),
                scores.get("monetary_score"),
                scores.get("inflation_score"),
                scores.get("market_score"),
                scores.get("primary_regime"),
                scores.get("cycle_phase"),
                scores.get("stress_index"),
                scores.get("confidence"),
                scores.get("binding_constraint"),
                fwd_returns.get("5d"),
                fwd_returns.get("20d"),
                fwd_returns.get("60d"),
                fwd_returns.get("max_dd_20d"),
                scores.get("modules_used", "full"),
                is_warmup,
                json.dumps(scores.get("trajectory_json")) if scores.get("trajectory_json") else None,
            )

            total_steps += 1
            if not is_warmup:
                scored_steps += 1

            if total_steps % 26 == 0:
                logger.info(f"  Progress: {current} ({total_steps} steps, {scored_steps} scored)")

        except Exception as e:
            logger.error(f"Error at {current}: {e}")
            import traceback
            traceback.print_exc()

        current += timedelta(days=step_days)

    logger.info(f"Backtest complete: {total_steps} total steps, {scored_steps} scored")
    return run_id
