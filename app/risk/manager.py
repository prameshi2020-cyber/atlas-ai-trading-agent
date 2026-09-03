from __future__ import annotations

import math
from datetime import datetime, timezone

from app.risk.models import RiskConfig, RiskDecision, RiskState
from app.risk.sizing import calculate_position_size
from app.database.models import (
    initialize_risk_state_database,
    load_risk_state,
    save_risk_state,
)

class RiskManager:
    def __init__(
        self,
        *,
        initial_equity: float,
        config: RiskConfig | None = None,
    ) -> None:
        if not math.isfinite(initial_equity) or initial_equity <= 0:
            raise ValueError("initial_equity must be greater than zero")

        self.config = config or RiskConfig()

        self.state = RiskState(
            equity_peak=initial_equity,
            day_start_equity=initial_equity,
        )

        initialize_risk_state_database()

    def _validate_config(self) -> None:
        if not 0 < self.config.risk_per_trade <= 1:
            raise ValueError(
                "risk_per_trade must be between 0 and 1"
            )

        if not 0 < self.config.daily_loss_limit_pct <= 1:
            raise ValueError(
                "daily_loss_limit_pct must be between 0 and 1"
            )

        if not 0 < self.config.max_drawdown_pct <= 1:
            raise ValueError(
                "max_drawdown_pct must be between 0 and 1"
            )

        if self.config.max_concurrent_positions < 1:
            raise ValueError(
                "max_concurrent_positions must be at least 1"
            )

        if self.config.maximum_position_pct <= 0:
            raise ValueError(
                "maximum_position_pct must be greater than zero"
            )
        
    def _update_day(self, timestamp: datetime, equity: float | None = None) -> None:
        timestamp = timestamp.astimezone(timezone.utc)
        day = timestamp.date().isoformat()

        if self.state.current_day is None:
            self.state.current_day = day
            return

        if day != self.state.current_day:
            self.state.current_day = day
            if equity is not None:
                self.state.day_start_equity = equity
            self.state.realized_daily_pnl = 0.0

    def update_equity(
        self,
        *,
        equity: float,
        timestamp: datetime,
    ) -> None:

        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("equity must be greater than zero")

        self._update_day(timestamp, equity)

        self.state.equity_peak = max(
            self.state.equity_peak,
            equity,
        )

        drawdown = (
            self.state.equity_peak - equity
        ) / self.state.equity_peak

        if drawdown >= self.config.max_drawdown_pct:
            self.state.halted = True
            self.state.halt_reason = "drawdown limit exceeded"

        self.save_state()

    def record_realized_pnl(
        self,
        *,
        pnl: float,
        timestamp: datetime,
    ) -> None:

        if not math.isfinite(pnl):
            raise ValueError("pnl must be finite")

        self._update_day(timestamp)

        self.state.realized_daily_pnl += pnl

        if self.state.day_start_equity <= 0:
            raise ValueError("day_start_equity must be greater than zero")

        daily_loss = (
            -self.state.realized_daily_pnl
            / self.state.day_start_equity
        )

        if daily_loss >= self.config.daily_loss_limit_pct:
            self.state.halted = True
            self.state.halt_reason = "daily loss limit exceeded"

        self.save_state()

    def register_position_open(self) -> bool:
        if (
            self.state.concurrent_positions
            >= self.config.max_concurrent_positions
        ):
            return False

        self.state.concurrent_positions += 1
        self.save_state()
        return True

    def restore_open_positions(self, count: int) -> None:
        if count < 0:
            raise ValueError(
                "count must be greater than or equal to zero"
            )

        if count > self.config.max_concurrent_positions:
            raise ValueError(
                "restored open positions exceed "
                "max_concurrent_positions"
            )

        self.state.concurrent_positions = count
        self.save_state()

    def register_position_close(self) -> None:
        if self.state.concurrent_positions > 0:
            self.state.concurrent_positions -= 1
            self.save_state()

    def evaluate_trade(
        self,
        *,
        equity: float,
        entry_price: float,
        stop_price: float,
        timestamp: datetime,
    ) -> RiskDecision:

        self._validate_config()
        self.update_equity(
            equity=equity,
            timestamp=timestamp,
        )

        if self.state.halted:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                risk_amount=0.0,
                position_notional=0.0,
                reason=self.state.halt_reason
                or "risk manager halted",
            )

        if (
            self.state.concurrent_positions
            >= self.config.max_concurrent_positions
        ):
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                risk_amount=0.0,
                position_notional=0.0,
                reason="maximum concurrent positions reached",
            )

        try:
            quantity, risk_amount, position_notional = (
                calculate_position_size(
                    equity=equity,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    risk_per_trade=self.config.risk_per_trade,
                    maximum_position_pct=self.config.maximum_position_pct,
                    minimum_position_size=self.config.minimum_position_size,
                )
            )
        except ValueError as exc:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                risk_amount=0.0,
                position_notional=0.0,
                reason=str(exc),
            )

        if quantity <= 0:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                risk_amount=risk_amount,
                position_notional=0.0,
                reason="position size below minimum",
            )

        return RiskDecision(
            allowed=True,
            quantity=quantity,
            risk_amount=risk_amount,
            position_notional=position_notional,
            reason="approved",
        )

    def reset_halt(self) -> None:
        self.state.halted = False
        self.state.halt_reason = None
        self.save_state()

    def restore_state(self, state: RiskState) -> None:
        self.state = RiskState(
            equity_peak=state.equity_peak,
            day_start_equity=state.day_start_equity,
            realized_daily_pnl=state.realized_daily_pnl,
            concurrent_positions=state.concurrent_positions,
            current_day=state.current_day,
            halted=state.halted,
            halt_reason=state.halt_reason,
        )

        self.save_state()

    def save_state(self) -> None:
        save_risk_state(
            equity_peak=self.state.equity_peak,
            day_start_equity=self.state.day_start_equity,
            realized_daily_pnl=self.state.realized_daily_pnl,
            concurrent_positions=self.state.concurrent_positions,
            current_day=self.state.current_day,
            halted=self.state.halted,
            halt_reason=self.state.halt_reason,
        )

    def load_state(self) -> bool:
        state = load_risk_state()

        if state is None:
            return False

        if state["equity_peak"] <= 0:
            raise ValueError(
                "persisted risk equity_peak must be greater than zero"
            )

        if state["day_start_equity"] <= 0:
            raise ValueError(
                "persisted risk day_start_equity must be greater than zero"
            )

        if state["concurrent_positions"] < 0:
            raise ValueError(
                "persisted risk concurrent_positions must be "
                "greater than or equal to zero"
            )

        if (
            state["concurrent_positions"]
            > self.config.max_concurrent_positions
        ):
            raise ValueError(
                "persisted risk concurrent_positions exceed "
                "max_concurrent_positions"
            )

        self.state = RiskState(
            equity_peak=state["equity_peak"],
            day_start_equity=state["day_start_equity"],
            realized_daily_pnl=state["realized_daily_pnl"],
            concurrent_positions=state["concurrent_positions"],
            current_day=state["current_day"],
            halted=state["halted"],
            halt_reason=state["halt_reason"],
        )

        return True