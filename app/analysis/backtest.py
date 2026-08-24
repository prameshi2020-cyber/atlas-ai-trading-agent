from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.analysis.strategy import StrategyEngine


@dataclass(slots=True)
class BacktestTrade:
    symbol: str
    side: str

    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    pnl_pct: float
    pnl_amount: float

    exit_reason: str


@dataclass(slots=True)
class BacktestResult:
    symbol: str

    initial_capital: float
    final_capital: float

    total_return_pct: float

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float

    max_drawdown_pct: float

    profit_factor: float
    expectancy_pct: float

    average_trade_pct: float
    average_win_pct: float
    average_loss_pct: float

    largest_win_pct: float
    largest_loss_pct: float

    long_trades: int
    long_wins: int
    short_trades: int
    short_wins: int

    trades: list[BacktestTrade]


class BacktestEngine:
    """
    Backtesting engine for ATLAS.

    Signals are generated from completed candles.
    Entries/exits execute at the next candle open.

    Supports:

    - Trading fees
    - Slippage
    - ATR-based stop loss
    - ATR-based take profit
    - Long/short positions
    - Equity tracking
    - Performance statistics
    """

    def __init__(
        self,
        strategy: StrategyEngine,
        initial_capital: float = 10_000.0,
        fee_pct: float = 0.10,
        slippage_pct: float = 0.05,
        stop_loss_atr: float | None = 1.5,
        take_profit_atr: float | None = 3.0,
    ) -> None:

        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        if fee_pct < 0:
            raise ValueError("fee_pct cannot be negative")

        if slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")

        if stop_loss_atr is not None and stop_loss_atr <= 0:
            raise ValueError("stop_loss_atr must be positive")

        if take_profit_atr is not None and take_profit_atr <= 0:
            raise ValueError("take_profit_atr must be positive")

        self.strategy = strategy
        self.initial_capital = initial_capital

        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr

    # ---------------------------------------------------------
    # Execution prices
    # ---------------------------------------------------------

    def _apply_entry_slippage(
        self,
        side: str,
        price: float,
    ) -> float:

        if side == "LONG":
            return price * (1 + self.slippage_pct / 100)

        return price * (1 - self.slippage_pct / 100)

    def _apply_exit_slippage(
        self,
        side: str,
        price: float,
    ) -> float:

        if side == "LONG":
            return price * (1 - self.slippage_pct / 100)

        return price * (1 + self.slippage_pct / 100)

    # ---------------------------------------------------------
    # Trade PnL
    # ---------------------------------------------------------

    def _calculate_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
    ) -> float:

        if side == "LONG":
            gross_return = (
                exit_price - entry_price
            ) / entry_price

        else:
            gross_return = (
                entry_price - exit_price
            ) / entry_price

        # Entry + exit fees.
        net_return = gross_return - (
            2 * self.fee_pct / 100
        )

        return net_return * 100

    # ---------------------------------------------------------
    # Run backtest
    # ---------------------------------------------------------

    def run(
        self,
        symbol: str,
        features: pd.DataFrame,
    ) -> BacktestResult:

        if features.empty:
            raise ValueError("Feature dataframe is empty")

        if len(features) < 3:
            raise ValueError("At least 3 rows are required")

        if "atr_14" not in features.columns:
            raise ValueError(
                "Feature dataframe must contain atr_14"
            )

        data = (
            features
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        capital = self.initial_capital

        position: str | None = None

        entry_price: float | None = None
        entry_time = None
        entry_atr: float | None = None

        trades: list[BacktestTrade] = []

        equity_curve: list[float] = [
            capital
        ]

        # -----------------------------------------------------
        # Main loop
        # -----------------------------------------------------

        for i in range(len(data) - 1):

            current = data.iloc[i]
            next_candle = data.iloc[i + 1]

            signal = self.strategy.generate_signal(
                symbol=symbol,
                features=data.iloc[: i + 1],
            )

            next_open = float(
                next_candle["open"]
            )

            high = float(
                next_candle["high"]
            )

            low = float(
                next_candle["low"]
            )

            atr = float(
                current["atr_14"]
            )

            # -------------------------------------------------
            # Manage existing position
            # -------------------------------------------------

            if position is not None:

                stop_price = None
                target_price = None

                if (
                    entry_price is not None
                    and entry_atr is not None
                ):

                    if position == "LONG":

                        if self.stop_loss_atr is not None:
                            stop_price = (
                                entry_price
                                - entry_atr
                                * self.stop_loss_atr
                            )

                        if self.take_profit_atr is not None:
                            target_price = (
                                entry_price
                                + entry_atr
                                * self.take_profit_atr
                            )

                    else:

                        if self.stop_loss_atr is not None:
                            stop_price = (
                                entry_price
                                + entry_atr
                                * self.stop_loss_atr
                            )

                        if self.take_profit_atr is not None:
                            target_price = (
                                entry_price
                                - entry_atr
                                * self.take_profit_atr
                            )

                exit_price = None
                exit_reason = None

                # ---------------------------------------------
                # LONG risk management
                # ---------------------------------------------

                if position == "LONG":

                    if (
                        stop_price is not None
                        and low <= stop_price
                    ):
                        exit_price = stop_price
                        exit_reason = "STOP_LOSS"

                    elif (
                        target_price is not None
                        and high >= target_price
                    ):
                        exit_price = target_price
                        exit_reason = "TAKE_PROFIT"

                # ---------------------------------------------
                # SHORT risk management
                # ---------------------------------------------

                else:

                    if (
                        stop_price is not None
                        and high >= stop_price
                    ):
                        exit_price = stop_price
                        exit_reason = "STOP_LOSS"

                    elif (
                        target_price is not None
                        and low <= target_price
                    ):
                        exit_price = target_price
                        exit_reason = "TAKE_PROFIT"

                # ---------------------------------------------
                # Close position due to SL/TP
                # ---------------------------------------------

                if exit_price is not None:

                    exit_price = self._apply_exit_slippage(
                        position,
                        exit_price,
                    )

                    pnl_pct = self._calculate_pnl(
                        position,
                        entry_price,
                        exit_price,
                    )

                    pnl_amount = (
                        capital
                        * pnl_pct
                        / 100
                    )

                    capital += pnl_amount

                    trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            side=position,
                            entry_time=entry_time,
                            exit_time=next_candle["timestamp"],
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_pct=pnl_pct,
                            pnl_amount=pnl_amount,
                            exit_reason=exit_reason,
                        )
                    )

                    position = None
                    entry_price = None
                    entry_time = None
                    entry_atr = None

                    equity_curve.append(capital)

                    continue

            # -------------------------------------------------
            # Signal-based position changes
            # -------------------------------------------------

            if signal.action == "BUY":

                # Open LONG
                if position is None:

                    position = "LONG"

                    entry_price = (
                        self._apply_entry_slippage(
                            "LONG",
                            next_open,
                        )
                    )

                    entry_time = (
                        next_candle["timestamp"]
                    )

                    entry_atr = atr

                # SHORT -> LONG
                elif position == "SHORT":

                    exit_price = (
                        self._apply_exit_slippage(
                            "SHORT",
                            next_open,
                        )
                    )

                    pnl_pct = self._calculate_pnl(
                        "SHORT",
                        entry_price,
                        exit_price,
                    )

                    pnl_amount = (
                        capital
                        * pnl_pct
                        / 100
                    )

                    capital += pnl_amount

                    trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            side="SHORT",
                            entry_time=entry_time,
                            exit_time=next_candle["timestamp"],
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_pct=pnl_pct,
                            pnl_amount=pnl_amount,
                            exit_reason="SIGNAL",
                        )
                    )

                    position = "LONG"

                    entry_price = (
                        self._apply_entry_slippage(
                            "LONG",
                            next_open,
                        )
                    )

                    entry_time = (
                        next_candle["timestamp"]
                    )

                    entry_atr = atr

            elif signal.action == "SELL":

                # Open SHORT
                if position is None:

                    position = "SHORT"

                    entry_price = (
                        self._apply_entry_slippage(
                            "SHORT",
                            next_open,
                        )
                    )

                    entry_time = (
                        next_candle["timestamp"]
                    )

                    entry_atr = atr

                # LONG -> SHORT
                elif position == "LONG":

                    exit_price = (
                        self._apply_exit_slippage(
                            "LONG",
                            next_open,
                        )
                    )

                    pnl_pct = self._calculate_pnl(
                        "LONG",
                        entry_price,
                        exit_price,
                    )

                    pnl_amount = (
                        capital
                        * pnl_pct
                        / 100
                    )

                    capital += pnl_amount

                    trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            side="LONG",
                            entry_time=entry_time,
                            exit_time=next_candle["timestamp"],
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_pct=pnl_pct,
                            pnl_amount=pnl_amount,
                            exit_reason="SIGNAL",
                        )
                    )

                    position = "SHORT"

                    entry_price = (
                        self._apply_entry_slippage(
                            "SHORT",
                            next_open,
                        )
                    )

                    entry_time = (
                        next_candle["timestamp"]
                    )

                    entry_atr = atr

            # -------------------------------------------------
            # Mark-to-market equity
            # -------------------------------------------------

            mark_price = float(
                next_candle["close"]
            )

            mark_equity = capital

            if (
                position is not None
                and entry_price is not None
            ):

                if position == "LONG":

                    unrealized = (
                        mark_price - entry_price
                    ) / entry_price

                else:

                    unrealized = (
                        entry_price - mark_price
                    ) / entry_price

                mark_equity = (
                    capital
                    * (1 + unrealized)
                )

            equity_curve.append(
                mark_equity
            )

        # -----------------------------------------------------
        # Close final position
        # -----------------------------------------------------

        if (
            position is not None
            and entry_price is not None
        ):

            final_price = float(
                data.iloc[-1]["close"]
            )

            final_time = data.iloc[-1][
                "timestamp"
            ]

            exit_price = (
                self._apply_exit_slippage(
                    position,
                    final_price,
                )
            )

            pnl_pct = self._calculate_pnl(
                position,
                entry_price,
                exit_price,
            )

            pnl_amount = (
                capital
                * pnl_pct
                / 100
            )

            capital += pnl_amount

            trades.append(
                BacktestTrade(
                    symbol=symbol,
                    side=position,
                    entry_time=entry_time,
                    exit_time=final_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_amount=pnl_amount,
                    exit_reason="END_OF_DATA",
                )
            )

            equity_curve.append(
                capital
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        total_trades = len(trades)

        winning_trades = sum(
            trade.pnl_pct > 0
            for trade in trades
        )

        losing_trades = sum(
            trade.pnl_pct <= 0
            for trade in trades
        )

        win_rate_pct = (
            winning_trades
            / total_trades
            * 100
            if total_trades
            else 0.0
        )

        pnls = [
            trade.pnl_pct
            for trade in trades
        ]

        winning_pnls = [
            pnl for pnl in pnls
            if pnl > 0
        ]

        losing_pnls = [
            pnl for pnl in pnls
            if pnl <= 0
        ]

        gross_profit = sum(
            winning_pnls
        )

        gross_loss = abs(
            sum(losing_pnls)
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = float("inf")

        average_trade_pct = (
            sum(pnls)
            / len(pnls)
            if pnls
            else 0.0
        )

        average_win_pct = (
            sum(winning_pnls)
            / len(winning_pnls)
            if winning_pnls
            else 0.0
        )

        average_loss_pct = (
            sum(losing_pnls)
            / len(losing_pnls)
            if losing_pnls
            else 0.0
        )

        largest_win_pct = (
            max(winning_pnls)
            if winning_pnls
            else 0.0
        )

        largest_loss_pct = (
            min(losing_pnls)
            if losing_pnls
            else 0.0
        )

        # -----------------------------------------------------
        # Expectancy
        # -----------------------------------------------------

        expectancy_pct = (
            (
                winning_trades
                / total_trades
                * average_win_pct
            )
            + (
                losing_trades
                / total_trades
                * average_loss_pct
            )
            if total_trades
            else 0.0
        )

        # -----------------------------------------------------
        # Long / Short statistics
        # -----------------------------------------------------

        long_trades = sum(
            trade.side == "LONG"
            for trade in trades
        )

        long_wins = sum(
            trade.side == "LONG"
            and trade.pnl_pct > 0
            for trade in trades
        )

        short_trades = sum(
            trade.side == "SHORT"
            for trade in trades
        )

        short_wins = sum(
            trade.side == "SHORT"
            and trade.pnl_pct > 0
            for trade in trades
        )

        # -----------------------------------------------------
        # Maximum drawdown
        # -----------------------------------------------------

        peak = equity_curve[0]
        max_drawdown = 0.0

        for equity in equity_curve:

            if equity > peak:
                peak = equity

            drawdown = (
                (equity - peak)
                / peak
                * 100
            )

            max_drawdown = min(
                max_drawdown,
                drawdown,
            )

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        total_return_pct = (
            (
                capital
                - self.initial_capital
            )
            / self.initial_capital
            * 100
        )

        return BacktestResult(
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_return_pct=total_return_pct,

            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate_pct,

            max_drawdown_pct=max_drawdown,

            profit_factor=profit_factor,
            expectancy_pct=expectancy_pct,

            average_trade_pct=average_trade_pct,
            average_win_pct=average_win_pct,
            average_loss_pct=average_loss_pct,

            largest_win_pct=largest_win_pct,
            largest_loss_pct=largest_loss_pct,

            long_trades=long_trades,
            long_wins=long_wins,

            short_trades=short_trades,
            short_wins=short_wins,

            trades=trades,
        )