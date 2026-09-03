from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.analysis.strategy import StrategyEngine
from app.analysis.indicators import REQUIRED_FEATURE_COLUMNS, drop_warmup_rows


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

    CAUSAL EXECUTION MODEL
    -----------------------
    For a decision made at candle i (closed):
        1. The strategy sees only data.iloc[: i + 1]      (candles <= i)
        2. Any resulting entry/exit executes at candle i+1's OPEN
        3. Candle i+1's HIGH/LOW/CLOSE are then used to:
             - evaluate whether a stop-loss / take-profit on any
               position (including one just opened this same bar)
               was touched during candle i+1, and
             - mark the position to market at candle i+1's CLOSE.
    Candle i+1 is fully known/closed by the time step 3 runs, so this
    does not introduce look-ahead: candle i+1's open was already being
    used for execution and its close was already being used for
    mark-to-market before this fix, so using its high/low for
    intrabar risk is consistent, not an extension into the future.

    ENTRY-BAR STOP-LOSS / TAKE-PROFIT
    -----------------------------------
    A position opened at candle N's open is evaluated against candle
    N's own high/low for a stop-loss or take-profit hit, in the same
    step it was opened. (Previously this engine only checked SL/TP
    starting from the bar *after* entry, meaning a stop breached
    within the entry candle itself was never detected -- a bug fixed
    in this version.)

    SAME-BAR STOP-LOSS / TAKE-PROFIT AMBIGUITY POLICY
    -----------------------------------------------------
    OHLC data alone cannot tell us whether price touched the
    stop-loss or the take-profit level *first* within a single
    candle when both levels fall inside that candle's [low, high]
    range. This engine resolves that ambiguity deterministically and
    conservatively: STOP-LOSS always takes priority over TAKE-PROFIT
    when both are touched in the same candle. This never assumes the
    more favorable outcome for the trader. See `_evaluate_exit`.
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
    # Stop-loss / take-profit evaluation
    # ---------------------------------------------------------

    def _evaluate_exit(
        self,
        position: str,
        entry_price: float | None,
        entry_atr: float | None,
        bar_high: float,
        bar_low: float,
    ) -> tuple[float | None, str | None]:
        """
        Check whether `position` (opened at `entry_price`, sized off
        `entry_atr`) had its stop-loss or take-profit touched within
        a single bar's [bar_low, bar_high] range.

        Returns (exit_price, exit_reason), or (None, None) if neither
        level was touched.

        A position can never be evaluated without a known ATR: if
        `entry_atr` is missing or NaN, this method returns (None,
        None) rather than guessing -- callers must not open a
        position without a real ATR in the first place (see `run`),
        so this is a defensive backstop, not the primary guard. A NaN
        ATR must never be silently treated as "no stop" while still
        allowing a position to trade.

        CONSERVATIVE SAME-BAR POLICY: if both the stop-loss and the
        take-profit level fall inside the same bar's range, OHLC data
        cannot tell us which was touched first intrabar. This method
        always resolves that ambiguity in favor of STOP_LOSS -- it
        never assumes the more favorable outcome for the trader.
        """
        if pd.isna(entry_price) or pd.isna(entry_atr):
            return None, None

        stop_price: float | None = None
        target_price: float | None = None

        if position == "LONG":
            if self.stop_loss_atr is not None:
                stop_price = entry_price - entry_atr * self.stop_loss_atr

            if self.take_profit_atr is not None:
                target_price = entry_price + entry_atr * self.take_profit_atr

            # Stop-loss checked first: conservative same-bar priority.
            if stop_price is not None and bar_low <= stop_price:
                return stop_price, "STOP_LOSS"

            if target_price is not None and bar_high >= target_price:
                return target_price, "TAKE_PROFIT"

        else:  # SHORT
            if self.stop_loss_atr is not None:
                stop_price = entry_price + entry_atr * self.stop_loss_atr

            if self.take_profit_atr is not None:
                target_price = entry_price - entry_atr * self.take_profit_atr

            # Stop-loss checked first: conservative same-bar priority.
            if stop_price is not None and bar_high >= stop_price:
                return stop_price, "STOP_LOSS"

            if target_price is not None and bar_low <= target_price:
                return target_price, "TAKE_PROFIT"

        return None, None

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

        missing_columns = set(REQUIRED_FEATURE_COLUMNS) - set(features.columns)

        if missing_columns:
            raise ValueError(
                f"Feature dataframe is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        data = (
            features
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # -----------------------------------------------------
        # Warm-up handling (see class docstring / indicators.py)
        # -----------------------------------------------------
        #
        # Rows where any required indicator is still NaN (the
        # rolling/EWM warm-up period -- e.g. the first ~49 rows for
        # sma_50) are dropped here, BEFORE the strategy ever sees
        # them. Without this, NaN comparisons inside the strategy
        # (e.g. `ema_9 > ema_21`) silently evaluate to False in
        # Python, fabricating a bearish signal out of "no data yet"
        # instead of correctly abstaining. This also guarantees
        # entry_atr below can never be NaN, so a stop-loss can never
        # be silently disabled.

        data = drop_warmup_rows(data)

        if len(data) < 3:
            raise ValueError(
                "At least 3 fully warmed-up rows are required after "
                "dropping indicator warm-up rows (need candles beyond "
                "the longest indicator lookback window, e.g. sma_50's "
                "50-candle window)"
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
        #
        # For iteration i:
        #   1. signal is generated from data.iloc[: i + 1]  (causal)
        #   2. any entry/exit/reversal executes at data.iloc[i+1]'s
        #      open
        #   3. data.iloc[i+1]'s high/low is checked for a stop-loss
        #      / take-profit hit against BOTH a pre-existing position
        #      AND a position opened/reversed into on this very
        #      iteration -- this is the entry-bar SL/TP fix.
        #   4. data.iloc[i+1]'s close marks the resulting position
        #      to market.

        for i in range(len(data) - 1):

            current = data.iloc[i]
            next_candle = data.iloc[i + 1]

            signal = self.strategy.generate_signal(
                symbol=symbol,
                features=data.iloc[: i + 1],
            )

            next_open = float(next_candle["open"])
            bar_high = float(next_candle["high"])
            bar_low = float(next_candle["low"])
            next_time = next_candle["timestamp"]

            # atr_14 is guaranteed non-NaN here: drop_warmup_rows()
            # already removed every row where it (or any other
            # required indicator) was missing.
            atr_value = current["atr_14"]
            atr = None if pd.isna(atr_value) else float(atr_value)

            # -------------------------------------------------
            # Step 1: manage a position opened on a PRIOR bar
            # against THIS bar's high/low.
            # -------------------------------------------------

            if position is not None:

                exit_price, exit_reason = self._evaluate_exit(
                    position,
                    entry_price,
                    entry_atr,
                    bar_high,
                    bar_low,
                )

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
                            exit_time=next_time,
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

                    # No new entry on the same bar that stopped us
                    # out, consistent with this engine's existing
                    # one-decision-per-bar convention.
                    continue

            # -------------------------------------------------
            # Step 2: signal-driven open / reverse. Entry (and a
            # reversal's exit) execute at next_open -- candle
            # i+1's open.
            #
            # A BUY/SELL signal with no valid ATR (atr is None)
            # is intentionally NOT traded: opening a position
            # without a real ATR would mean an untracked,
            # unprotected stop-loss. In the normal pipeline this
            # branch is unreachable because drop_warmup_rows()
            # already guarantees atr_14 is always valid here; it
            # exists as a defensive backstop.
            # -------------------------------------------------

            if signal.action == "BUY":

                # Open LONG
                if position is None:

                    if atr is not None:

                        position = "LONG"

                        entry_price = (
                            self._apply_entry_slippage(
                                "LONG",
                                next_open,
                            )
                        )

                        entry_time = next_time
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
                            exit_time=next_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_pct=pnl_pct,
                            pnl_amount=pnl_amount,
                            exit_reason="SIGNAL",
                        )
                    )

                    if atr is not None:

                        position = "LONG"

                        entry_price = (
                            self._apply_entry_slippage(
                                "LONG",
                                next_open,
                            )
                        )

                        entry_time = next_time
                        entry_atr = atr

                    else:

                        position = None
                        entry_price = None
                        entry_time = None
                        entry_atr = None

            elif signal.action == "SELL":

                # Open SHORT
                if position is None:

                    if atr is not None:

                        position = "SHORT"

                        entry_price = (
                            self._apply_entry_slippage(
                                "SHORT",
                                next_open,
                            )
                        )

                        entry_time = next_time
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
                            exit_time=next_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_pct=pnl_pct,
                            pnl_amount=pnl_amount,
                            exit_reason="SIGNAL",
                        )
                    )

                    if atr is not None:

                        position = "SHORT"

                        entry_price = (
                            self._apply_entry_slippage(
                                "SHORT",
                                next_open,
                            )
                        )

                        entry_time = next_time
                        entry_atr = atr

                    else:

                        position = None
                        entry_price = None
                        entry_time = None
                        entry_atr = None

            # -------------------------------------------------
            # Step 3: THE FIX -- if a position was opened or
            # reversed into on THIS bar (step 2), check the SAME
            # bar's high/low for an entry-bar stop-loss / take
            # -profit hit. `entry_time == next_time` identifies a
            # position that was just opened this iteration (a
            # position carried over from a prior bar would have
            # an earlier entry_time, and was already handled --
            # or closed -- in step 1 above).
            # -------------------------------------------------

            if position is not None and entry_time == next_time:

                exit_price, exit_reason = self._evaluate_exit(
                    position,
                    entry_price,
                    entry_atr,
                    bar_high,
                    bar_low,
                )

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
                            exit_time=next_time,
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

            # -------------------------------------------------
            # Step 4: mark-to-market equity
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

            # positive magnitude: percentage distance from the most
            # recently observed peak down to the current equity. This
            # yields 0.0 for no drawdown and a positive value for a
            # real drawdown (e.g., 10.0 for a 10% drop).
            drawdown = ((peak - equity) / peak) * 100

            if drawdown > max_drawdown:
                max_drawdown = drawdown

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