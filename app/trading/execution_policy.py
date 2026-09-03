from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    """
    Shared execution contract between backtest and paper trading.

    The domain rules remain identical across environments: risk accounting,
    realized PnL, position direction, trade lifecycle, trade_id persistence,
    and deterministic stop/exit rules. The pricing source and market-event
    plumbing are environment-specific.
    """

    decision_timestamp: str = "decision candle close"
    execution_timestamp: str = "next market event or next candle open"
    execution_price_source: str = "environment-specific observed price"
    entry_semantics: str = "open on execution event; shared accounting rules"
    exit_semantics: str = "close on stop / take profit / reversal / manual"
    reversal_semantics: str = "close prior position before opening new direction"
    fee_semantics: str = "shared fee and slippage accounting in ExecutionEngine"
    slippage_semantics: str = "environment-specific plumbing, common accounting interface"
    stop_loss_semantics: str = "deterministic stop policy on observed price range"
