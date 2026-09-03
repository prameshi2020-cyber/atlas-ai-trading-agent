import pytest

from app.strategy.stop_policy import StopPolicy



def test_stop_trigger_rejects_invalid_candle_range():
    policy = StopPolicy(atr_multiplier=1.5)

    with pytest.raises(ValueError):
        policy.is_triggered(
            side="LONG",
            stop_price=95.0,
            high=94.0,
            low=95.0,
        )

def test_stop_policy_calculates_long_stop_from_atr():
    policy = StopPolicy(
        atr_multiplier=1.5,
    )

    stop_price = policy.calculate(
        side="LONG",
        entry_price=120.0,
        atr=9.333333333333334,
    )

    assert stop_price == pytest.approx(106.0)


def test_stop_policy_calculates_short_stop_from_atr():
    policy = StopPolicy(
        atr_multiplier=1.5,
    )

    stop_price = policy.calculate(
        side="SHORT",
        entry_price=120.0,
        atr=9.333333333333334,
    )

    assert stop_price == pytest.approx(134.0)

def test_stop_policy_rejects_invalid_multiplier():
    with pytest.raises(
        ValueError,
        match="atr_multiplier must be greater than zero",
    ):
        StopPolicy(atr_multiplier=0.0)

    with pytest.raises(
        ValueError,
        match="atr_multiplier must be greater than zero",
    ):
        StopPolicy(atr_multiplier=-1.0)


def test_stop_policy_rejects_invalid_side():
    policy = StopPolicy()

    with pytest.raises(
        ValueError,
        match="side must be LONG or SHORT",
    ):
        policy.calculate(
            side="FLAT",
            entry_price=120.0,
            atr=10.0,
        )


def test_stop_policy_rejects_invalid_entry_price():
    policy = StopPolicy()

    with pytest.raises(
        ValueError,
        match="entry_price must be greater than zero",
    ):
        policy.calculate(
            side="LONG",
            entry_price=0.0,
            atr=10.0,
        )

    with pytest.raises(
        ValueError,
        match="entry_price must be greater than zero",
    ):
        policy.calculate(
            side="LONG",
            entry_price=-100.0,
            atr=10.0,
        )


def test_stop_policy_rejects_invalid_atr():
    policy = StopPolicy()

    with pytest.raises(
        ValueError,
        match="atr must be greater than zero",
    ):
        policy.calculate(
            side="LONG",
            entry_price=120.0,
            atr=0.0,
        )

    with pytest.raises(
        ValueError,
        match="atr must be greater than zero",
    ):
        policy.calculate(
            side="LONG",
            entry_price=120.0,
            atr=-5.0,
        )


def test_stop_policy_rejects_stop_price_below_or_equal_to_zero():
    policy = StopPolicy(
        atr_multiplier=2.0,
    )

    with pytest.raises(
        ValueError,
        match="calculated stop_price must be greater than zero",
    ):
        policy.calculate(
            side="LONG",
            entry_price=10.0,
            atr=10.0,
        )

def test_long_stop_is_triggered_when_candle_low_reaches_stop():
    policy = StopPolicy(atr_multiplier=1.5)

    assert policy.is_triggered(
        side="LONG",
        stop_price=95.0,
        high=101.0,
        low=95.0,
    )


def test_long_stop_is_triggered_when_candle_low_breaks_stop():
    policy = StopPolicy(atr_multiplier=1.5)

    assert policy.is_triggered(
        side="LONG",
        stop_price=95.0,
        high=101.0,
        low=94.0,
    )


def test_long_stop_is_not_triggered_above_stop():
    policy = StopPolicy(atr_multiplier=1.5)

    assert not policy.is_triggered(
        side="LONG",
        stop_price=95.0,
        high=101.0,
        low=96.0,
    )


def test_short_stop_is_triggered_when_candle_high_reaches_stop():
    policy = StopPolicy(atr_multiplier=1.5)

    assert policy.is_triggered(
        side="SHORT",
        stop_price=105.0,
        high=105.0,
        low=99.0,
    )


def test_short_stop_is_triggered_when_candle_high_breaks_stop():
    policy = StopPolicy(atr_multiplier=1.5)

    assert policy.is_triggered(
        side="SHORT",
        stop_price=105.0,
        high=106.0,
        low=99.0,
    )


def test_short_stop_is_not_triggered_below_stop():
    policy = StopPolicy(atr_multiplier=1.5)

    assert not policy.is_triggered(
        side="SHORT",
        stop_price=105.0,
        high=104.0,
        low=99.0,
    )