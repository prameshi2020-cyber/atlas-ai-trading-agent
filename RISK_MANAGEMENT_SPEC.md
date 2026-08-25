# ATLAS Risk Management Specification

## Goal

Implement a deterministic, configurable risk-management layer for ATLAS.

The risk layer must control position sizing and trading limits without changing
the existing StrategyEngine signal generation logic.

The risk-management layer must be independently testable.

---

# 1. Risk-Based Position Sizing

The system must NOT allocate 100% of account equity to every trade.

Position sizing must be based on:

    risk_amount = equity * risk_per_trade

    stop_distance = abs(entry_price - stop_price)

    position_size = risk_amount / stop_distance

The resulting position size represents the quantity of the traded asset.

The system must support configurable:

- risk_per_trade
- minimum_position_size
- maximum_position_pct

Default configuration:

    risk_per_trade = 0.01
    maximum_position_pct = 1.00

This means:

- maximum intended loss per trade = 1% of equity
- position notional cannot exceed 100% of equity

The sizing function must reject invalid inputs:

- equity <= 0
- entry_price <= 0
- stop_price <= 0
- stop_distance <= 0
- non-finite values

---

# 2. Maximum Drawdown Protection

The risk manager must support a maximum drawdown circuit breaker.

Default:

    max_drawdown_pct = 0.10

If equity falls 10% or more from the historical equity peak:

    trading_allowed = False

The circuit breaker must remain active until explicitly reset.

The risk manager must never automatically resume trading merely because
equity temporarily recovers.

---

# 3. Daily Loss Limit

The risk manager must support a daily loss limit.

Default:

    daily_loss_limit_pct = 0.03

If realized daily PnL reaches -3% of the day's starting equity:

    trading_allowed = False

The limit resets only when a new trading day begins.

The daily loss calculation must use realized PnL, not unrealized PnL.

---

# 4. Maximum Concurrent Positions

The system currently operates one symbol at a time, but the risk layer
must support a configurable maximum number of concurrent positions.

Default:

    max_concurrent_positions = 1

If the limit is reached:

    new position = rejected

---

# 5. Position Notional Limit

Even when risk-based sizing produces a large position, the risk manager
must enforce:

    position_notional <= equity * maximum_position_pct

Default:

    maximum_position_pct = 1.00

Therefore a single position cannot exceed 100% of current equity.

---

# 6. Risk Decision Object

The risk layer should return a structured decision rather than only
a boolean.

Example conceptual structure:

    RiskDecision(
        allowed=True/False,
        quantity=...,
        risk_amount=...,
        position_notional=...,
        reason="..."
    )

When rejected, `quantity` must be zero.

The rejection reason must be explicit.

Examples:

    "drawdown limit exceeded"
    "daily loss limit exceeded"
    "maximum concurrent positions reached"
    "invalid stop distance"
    "position size below minimum"
    "position notional exceeds maximum"

---

# 7. Separation of Responsibilities

StrategyEngine remains responsible for:

- BUY
- SELL
- HOLD
- strategy score
- strategy confidence
- signal reasons

RiskManager is responsible for:

- whether a trade is allowed
- position size
- risk amount
- drawdown limits
- daily loss limits
- concurrent position limits
- notional limits

BacktestEngine is responsible for:

- candle iteration
- execution
- SL/TP handling
- PnL
- equity curve
- trade records

Do not move strategy scoring logic into the risk layer.

Do not modify StrategyEngine during this milestone.

---

# 8. Backtest Metrics

BacktestResult should eventually expose:

- total return
- max drawdown
- win rate
- profit factor
- expectancy
- average trade
- average win
- average loss
- largest win
- largest loss
- total fees
- total slippage cost
- buy & hold return
- Sharpe ratio
- Sortino ratio
- maximum consecutive wins
- maximum consecutive losses

Metrics must be calculated deterministically from the trade/equity history.

---

# 9. No Strategy Tuning

This milestone must NOT modify:

- strategy scoring weights
- BUY threshold
- SELL threshold
- RSI thresholds
- EMA conditions
- MACD conditions
- momentum conditions
- signal confidence logic

The purpose of this milestone is infrastructure and risk control,
not optimization.

---

# 10. Testing Requirements

Tests must cover at minimum:

1. 1% risk sizing produces correct quantity.
2. Long position sizing.
3. Short position sizing.
4. Invalid stop distance is rejected.
5. Zero/negative equity is rejected.
6. Position notional cap is enforced.
7. Maximum drawdown circuit breaker activates.
8. Drawdown circuit breaker remains active.
9. Daily loss limit activates.
10. Daily loss resets on a new trading day.
11. Concurrent position limit is enforced.
12. Rejected risk decision returns quantity = 0.
13. Risk amount is deterministic.
14. Same input always produces same sizing.
15. Existing 24 deterministic backtest tests remain passing.

---

# 11. Important Constraint

Do NOT optimize the parameters using the current 500-candle datasets.

The following values are implementation defaults, not optimized values:

    risk_per_trade = 1%
    daily_loss_limit = 3%
    max_drawdown = 10%
    max_concurrent_positions = 1
    maximum_position_pct = 100%

These values must be configurable.

Parameter optimization will be performed only after
a substantially larger multi-regime dataset is available.