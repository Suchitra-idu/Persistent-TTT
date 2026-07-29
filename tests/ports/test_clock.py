from __future__ import annotations

import re

import pytest

from ttt.adapters.fake_clock import FakeClock
from ttt.adapters.system_clock import SystemClock
from ttt.ports.clock import STAMP_LENGTH, Clock

STAMP_PATTERN = re.compile(r"^\d{4}-\d{4}$")


class ClockConformance:
    @pytest.fixture
    def clock(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, clock):
        assert isinstance(clock, Clock)

    def test_monotonic_never_goes_backwards(self, clock):
        readings = [clock.monotonic() for _ in range(5)]

        assert readings == sorted(readings)

    def test_the_stamp_is_month_day_hour_minute(self, clock):
        assert STAMP_PATTERN.match(clock.stamp())

    def test_the_stamp_has_the_declared_length(self, clock):
        assert len(clock.stamp()) == STAMP_LENGTH


class TestSystemClock(ClockConformance):
    @pytest.fixture
    def clock(self):
        return SystemClock()


class TestFakeClock(ClockConformance):
    @pytest.fixture
    def clock(self):
        return FakeClock()

    def test_time_moves_only_when_advanced(self, clock):
        before = clock.monotonic()
        clock.advance(2.5)

        assert (before, clock.monotonic()) == (0.0, 2.5)

    def test_it_refuses_to_go_backwards(self, clock):
        with pytest.raises(ValueError, match="does not go backwards"):
            clock.advance(-1.0)
