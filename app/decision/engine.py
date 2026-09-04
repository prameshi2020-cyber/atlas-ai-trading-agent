from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.analysis.research import ResearchBatchResult
from app.decision.models import DeploymentDecision, DeploymentVerdict, PolicyCheck
from app.decision.policy import DeploymentPolicy


def _utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _verdict(checks: tuple[PolicyCheck, ...]) -> DeploymentVerdict:
    failed = [c for c in checks if not c.passed]
    if not failed:
        return "APPROVE"
    if any(c.failure_verdict == "REJECT" for c in failed):
        return "REJECT"
    return "HOLD"


class DecisionEngine:
    """
    Turn a ResearchBatchResult into a DeploymentDecision by comparing already-computed
    metrics against a DeploymentPolicy.

    Pure and side-effect free: no clock, no I/O, no trading, risk or AI imports.
    Quality failures REJECT; operational failures HOLD; REJECT outranks HOLD.
    """

    def __init__(self, policy: DeploymentPolicy | None = None) -> None:
        if policy is not None and not isinstance(policy, DeploymentPolicy):
            raise TypeError("policy must be a DeploymentPolicy")
        self._policy = policy or DeploymentPolicy()

    @property
    def policy(self) -> DeploymentPolicy:
        return self._policy

    def decide(self, batch: ResearchBatchResult, *, now: datetime) -> DeploymentDecision:
        if not isinstance(batch, ResearchBatchResult):
            raise TypeError("batch must be a ResearchBatchResult")
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")

        policy = self._policy
        dataset = batch.request.dataset
        checks: list[PolicyCheck] = []

        # --- operational (HOLD) ---------------------------------------------------
        selected_id = batch.selected_candidate_id
        checks.append(
            PolicyCheck(
                name="selected_candidate",
                passed=selected_id is not None,
                failure_verdict="HOLD",
                observed=selected_id,
                threshold="required",
                detail="no candidate was selected by the research pipeline"
                if selected_id is None
                else f"candidate {selected_id!r} selected",
            )
        )

        if policy.require_no_batch_failures:
            failures = batch.batch_failures
            stages = sorted({str(f.stage) for f in failures})
            checks.append(
                PolicyCheck(
                    name="batch_failures",
                    passed=not failures,
                    failure_verdict="HOLD",
                    observed=len(failures),
                    threshold=0,
                    detail=f"{len(failures)} batch-level failure(s) at stage(s) {stages}" if failures else "no batch failures",
                )
            )

        if policy.maximum_dataset_age is not None:
            age = (_utc(now) - _utc(dataset.end)).to_pytimedelta()
            limit = policy.maximum_dataset_age
            checks.append(
                PolicyCheck(
                    name="dataset_fresh",
                    passed=age <= limit,
                    failure_verdict="HOLD",
                    observed=age,
                    threshold=limit,
                    detail=f"dataset age {age} exceeds {limit}" if age > limit else f"dataset age {age} within {limit}",
                )
            )

        if policy.allowed_symbols is not None:
            checks.append(
                PolicyCheck(
                    name="symbol_allowed",
                    passed=dataset.symbol in policy.allowed_symbols,
                    failure_verdict="HOLD",
                    observed=dataset.symbol,
                    threshold=sorted(policy.allowed_symbols),
                    detail=f"symbol {dataset.symbol!r} is not in the allowed set",
                )
            )

        if policy.allowed_timeframes is not None:
            checks.append(
                PolicyCheck(
                    name="timeframe_allowed",
                    passed=dataset.timeframe in policy.allowed_timeframes,
                    failure_verdict="HOLD",
                    observed=dataset.timeframe,
                    threshold=sorted(policy.allowed_timeframes),
                    detail=f"timeframe {dataset.timeframe!r} is not in the allowed set",
                )
            )

        # --- quality (REJECT) — only meaningful when a candidate exists -----------
        strategy_id: str | None = None
        parameters: tuple[tuple[str, object], ...] = ()

        if selected_id is not None:
            item = batch.result_for(selected_id)
            score = batch.score_for(selected_id)
            if item is None or score is None:
                raise ValueError(f"batch is inconsistent: selected candidate {selected_id!r} has no result/score")

            strategy_id = item.candidate.strategy_id
            parameters = tuple(item.candidate.parameters)
            bt = item.backtest_result
            wf = item.walk_forward_result

            checks.extend(
                [
                    PolicyCheck(
                        name="evaluation_pass",
                        passed=str(score.evaluation_decision) == "PASS",
                        failure_verdict="REJECT",
                        observed=str(score.evaluation_decision),
                        threshold="PASS",
                        detail=f"evaluation decision is {score.evaluation_decision}",
                    ),
                    PolicyCheck(
                        name="robustness_pass",
                        passed=str(score.robustness_decision) == "PASS",
                        failure_verdict="REJECT",
                        observed=str(score.robustness_decision),
                        threshold="PASS",
                        detail=f"robustness decision is {score.robustness_decision}",
                    ),
                    PolicyCheck(
                        name="walk_forward_windows",
                        passed=wf.total_windows >= policy.minimum_walk_forward_windows,
                        failure_verdict="REJECT",
                        observed=wf.total_windows,
                        threshold=policy.minimum_walk_forward_windows,
                        detail=f"{wf.total_windows} walk-forward window(s), minimum {policy.minimum_walk_forward_windows}",
                    ),
                    PolicyCheck(
                        name="oos_pass_rate",
                        passed=wf.oos_pass_rate_pct >= policy.minimum_oos_pass_rate_pct,
                        failure_verdict="REJECT",
                        observed=wf.oos_pass_rate_pct,
                        threshold=policy.minimum_oos_pass_rate_pct,
                        detail=f"OOS pass rate {wf.oos_pass_rate_pct:.2f}%, minimum {policy.minimum_oos_pass_rate_pct:.2f}%",
                    ),
                    PolicyCheck(
                        name="max_drawdown",
                        passed=bt.max_drawdown_pct <= policy.maximum_drawdown_pct,
                        failure_verdict="REJECT",
                        observed=bt.max_drawdown_pct,
                        threshold=policy.maximum_drawdown_pct,
                        detail=f"max drawdown {bt.max_drawdown_pct:.4f}%, maximum {policy.maximum_drawdown_pct:.4f}%",
                    ),
                    PolicyCheck(
                        name="total_trades",
                        passed=bt.total_trades >= policy.minimum_total_trades,
                        failure_verdict="REJECT",
                        observed=bt.total_trades,
                        threshold=policy.minimum_total_trades,
                        detail=f"{bt.total_trades} trade(s), minimum {policy.minimum_total_trades}",
                    ),
                ]
            )

        frozen_checks = tuple(checks)
        return DeploymentDecision(
            verdict=_verdict(frozen_checks),
            candidate_id=selected_id,
            strategy_id=strategy_id,
            parameters=parameters,
            symbol=dataset.symbol,
            timeframe=dataset.timeframe,
            reasons=tuple(c.detail for c in frozen_checks if not c.passed),
            checks=frozen_checks,
            research_fingerprint=batch.request.fingerprint,
            configuration_fingerprint=batch.request.configuration_fingerprint,
            dataset_fingerprint=dataset.fingerprint,
            decided_at=now,
        )