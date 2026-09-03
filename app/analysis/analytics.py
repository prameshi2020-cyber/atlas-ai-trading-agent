from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from app.trading.journal import load_all_trades


VALID_SIDES = {"LONG", "SHORT"}


@dataclass(frozen=True, slots=True)
class TradePerformance:
    trade_id: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    total_quantity: float
    total_pnl: float
    total_fee: float
    exit_count: int
    holding_time_seconds: float


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float

    gross_profit: float
    gross_loss: float
    net_pnl: float
    profit_factor: float
    expectancy: float

    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    long_trades: int
    long_wins: int
    short_trades: int
    short_wins: int

    partial_exit_count: int
    average_holding_time_seconds: float
    total_fees: float

    trades: list[TradePerformance]


class AnalyticsEngine:
    """
    Analyze completed persisted trade journal records.

    A single trade lifecycle may span multiple journal rows when partial
    exits occur. The analytics engine groups rows by trade_id, treating
    each non-NULL trade_id as one lifecycle. Legacy NULL trade_id rows are
    treated as distinct lifecycles so unrelated historic trades are not
    silently merged.
    """

    def _validate_row(self, row: dict) -> None:
        trade_id = row.get("trade_id")
        if trade_id is not None and not isinstance(trade_id, str):
            raise ValueError("trade_id must be a string or NULL")

        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            raise ValueError("symbol cannot be empty")

        side = str(row.get("side", "")).upper()
        if side not in VALID_SIDES:
            raise ValueError(f"invalid side: {side!r}")

        quantity = float(row.get("quantity"))
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("quantity must be positive")

        entry_price = float(row.get("entry_price"))
        exit_price = float(row.get("exit_price"))
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if not math.isfinite(exit_price) or exit_price <= 0:
            raise ValueError("exit_price must be positive")

        fee = float(row.get("fee"))
        if not math.isfinite(fee) or fee < 0:
            raise ValueError("fee must be finite and non-negative")

        entry_time_raw = row.get("entry_time")
        exit_time_raw = row.get("exit_time")
        if not isinstance(entry_time_raw, str) or not entry_time_raw:
            raise ValueError("entry_time must be a non-empty ISO string")
        if not isinstance(exit_time_raw, str) or not exit_time_raw:
            raise ValueError("exit_time must be a non-empty ISO string")

        try:
            entry_time = datetime.fromisoformat(entry_time_raw)
            exit_time = datetime.fromisoformat(exit_time_raw)
        except ValueError as exc:
            raise ValueError("timestamps must be valid ISO datetimes") from exc

        if exit_time < entry_time:
            raise ValueError("exit_time cannot be earlier than entry_time")

        pnl = float(row.get("pnl"))
        if not math.isfinite(pnl):
            raise ValueError("pnl must be finite")

    def _coerce_rows(self, rows: list[dict]) -> list[dict]:
        cleaned: list[dict] = []
        for row in rows:
            self._validate_row(row)
            cleaned.append(row)
        return cleaned

    def _group_lifecycles(self, rows: list[dict]) -> dict[str | None, list[dict]]:
        grouped: dict[str | None, list[dict]] = {}
        for row in rows:
            key = row.get("trade_id")
            if key is None:
                key = f"__legacy_null__:{len(grouped)}:{row.get('id')}"
            grouped.setdefault(key, []).append(row)
        return grouped

    def _build_trade_performance(self, trade_id: str | None, rows: list[dict]) -> TradePerformance:
        if not rows:
            raise ValueError("cannot build trade performance from empty rows")

        ordered = sorted(rows, key=lambda row: datetime.fromisoformat(row["entry_time"]))

        total_quantity = sum(float(row["quantity"]) for row in ordered)
        total_pnl = sum(float(row["pnl"]) for row in ordered)
        total_fee = sum(float(row["fee"]) for row in ordered)
        side_values = {str(row["side"]).upper() for row in ordered}
        if len(side_values) != 1:
            raise ValueError(f"inconsistent sides for trade lifecycle {trade_id!r}")

        side = next(iter(side_values))
        entry_time = min(datetime.fromisoformat(row["entry_time"]) for row in ordered)
        exit_time = max(datetime.fromisoformat(row["exit_time"]) for row in ordered)
        holding_time_seconds = (exit_time - entry_time).total_seconds()

        lifecyle_trade_id = trade_id if trade_id is not None else f"legacy_null_{entry_time.isoformat()}_{exit_time.isoformat()}"

        return TradePerformance(
            trade_id=lifecyle_trade_id,
            symbol=str(ordered[0]["symbol"]).strip(),
            side=side,
            entry_time=entry_time,
            exit_time=exit_time,
            total_quantity=total_quantity,
            total_pnl=total_pnl,
            total_fee=total_fee,
            exit_count=len(ordered),
            holding_time_seconds=holding_time_seconds,
        )

    def build_report(self) -> PerformanceReport:
        rows = self._coerce_rows(load_all_trades())
        grouped = self._group_lifecycles(rows)

        trades: list[TradePerformance] = []
        for trade_key, group_rows in grouped.items():
            trade = self._build_trade_performance(trade_key, group_rows)
            trades.append(trade)

        total_trades = len(trades)
        if total_trades == 0:
            zeros = PerformanceReport(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                net_pnl=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                long_trades=0,
                long_wins=0,
                short_trades=0,
                short_wins=0,
                partial_exit_count=0,
                average_holding_time_seconds=0.0,
                total_fees=0.0,
                trades=[],
            )
            return zeros

        winning_trades = sum(1 for trade in trades if trade.total_pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.total_pnl <= 0)
        win_rate_pct = (winning_trades / total_trades) * 100.0 if total_trades else 0.0

        gross_profit = sum(trade.total_pnl for trade in trades if trade.total_pnl > 0)
        gross_loss = abs(sum(trade.total_pnl for trade in trades if trade.total_pnl <= 0))
        net_pnl = sum(trade.total_pnl for trade in trades)
        if gross_loss == 0:
            profit_factor = float("inf") if winning_trades > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        expectancy = net_pnl / total_trades if total_trades else 0.0

        winning_values = [trade.total_pnl for trade in trades if trade.total_pnl > 0]
        losing_values = [trade.total_pnl for trade in trades if trade.total_pnl <= 0]
        average_win = (sum(winning_values) / len(winning_values)) if winning_values else 0.0
        average_loss = (sum(losing_values) / len(losing_values)) if losing_values else 0.0
        largest_win = max(winning_values) if winning_values else 0.0
        largest_loss = min(losing_values) if losing_values else 0.0

        long_trades = sum(1 for trade in trades if trade.side == "LONG")
        long_wins = sum(1 for trade in trades if trade.side == "LONG" and trade.total_pnl > 0)
        short_trades = sum(1 for trade in trades if trade.side == "SHORT")
        short_wins = sum(1 for trade in trades if trade.side == "SHORT" and trade.total_pnl > 0)

        partial_exit_count = sum(max(trade.exit_count - 1, 0) for trade in trades)
        average_holding_time_seconds = (
            sum(trade.holding_time_seconds for trade in trades) / total_trades if total_trades else 0.0
        )
        total_fees = sum(trade.total_fee for trade in trades)

        report = PerformanceReport(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate_pct,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_pnl=net_pnl,
            profit_factor=profit_factor,
            expectancy=expectancy,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            long_trades=long_trades,
            long_wins=long_wins,
            short_trades=short_trades,
            short_wins=short_wins,
            partial_exit_count=partial_exit_count,
            average_holding_time_seconds=average_holding_time_seconds,
            total_fees=total_fees,
            trades=trades,
        )
        return report
