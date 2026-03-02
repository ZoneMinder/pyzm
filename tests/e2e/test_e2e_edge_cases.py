"""E2E tests for ZM API edge cases, type coercion, and quirks.

These tests validate behaviors that are easy to get wrong with mocked responses:
type coercion mismatches, pagination coherence, and non-JSON response handling.
"""

import pytest

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Readonly tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_readonly
class TestEventsPagination:
    """Verify events pagination count vs list length coherence."""

    def test_count_gte_list_length(self, zm_api_live):
        """count() returns total matching events; list() is capped by max_events."""
        events = zm_api_live.events({"max_events": 3})
        count = events.count()
        length = len(events.list())
        assert isinstance(count, int)
        assert isinstance(length, int)
        # list length should not exceed requested max
        assert length <= 3
        # count is total matching, should be >= what we got
        assert count >= length


@pytest.mark.e2e_readonly
class TestTimezoneConsistency:
    """Verify timezone is consistent across calls."""

    def test_tz_same_across_calls(self, zm_api_live):
        tz1 = zm_api_live.tz()
        tz2 = zm_api_live.tz()
        assert tz1 == tz2


@pytest.mark.e2e_readonly
class TestVersionParseable:
    """Verify version fields can be parsed as integers."""

    def test_api_version_parts_are_ints(self, zm_api_live):
        v = zm_api_live.version()
        parts = v["api_version"].split(".")
        for part in parts:
            int(part)  # Should not raise

    def test_zm_version_parts_are_ints(self, zm_api_live):
        v = zm_api_live.version()
        parts = v["zm_version"].split(".")
        for part in parts:
            int(part)  # Should not raise


@pytest.mark.e2e_readonly
class TestMonitorIdTypeCoercion:
    """Verify monitor ID type coercion: accessor always returns int."""

    def test_round_trip_id(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        assert int(mon.get()["Id"]) == mon.id()


@pytest.mark.e2e_readonly
class TestEventScoreTypeCoercion:
    """Verify event score type coercion matches raw data."""

    def test_score_float_matches_raw(self, zm_api_live):
        events = zm_api_live.events({"max_events": 1})
        lst = events.list()
        if not lst:
            pytest.skip("No events available")
        ev = lst[0]
        raw = ev.get()
        score = ev.score()
        assert score["total"] == float(raw["TotScore"])
        assert score["average"] == float(raw["AvgScore"])
        assert score["max"] == float(raw["MaxScore"])


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_write
class TestArmDisarm:
    """Test arm/disarm handles non-JSON responses gracefully."""

    def test_arm_disarm(self, zm_api_live, e2e_monitor_factory, requires_write):
        mon, _ = e2e_monitor_factory(name="arm_test", function="Monitor")
        assert mon is not None
        # arm() may return dict or None (ZM may return non-JSON)
        result = mon.arm()
        assert result is None or isinstance(result, dict)
        # disarm()
        result = mon.disarm()
        assert result is None or isinstance(result, dict)


