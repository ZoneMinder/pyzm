"""E2E tests for Events and Event objects against a live ZoneMinder.

Readonly tests verify list/filter/accessors. Write tests delete events.
"""

import pytest

from pyzm.helpers.Events import Events
from pyzm.helpers.Event import Event

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Readonly tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_readonly
class TestEventList:
    """Verify events() returns a valid Events collection."""

    def test_returns_events_instance(self, zm_api_live):
        result = zm_api_live.events()
        assert isinstance(result, Events)

    def test_list_returns_list(self, zm_api_live):
        events = zm_api_live.events()
        lst = events.list()
        assert isinstance(lst, list)

    def test_items_are_event_objects(self, zm_api_live):
        events = zm_api_live.events({"max_events": 5})
        for ev in events.list():
            assert isinstance(ev, Event)

    def test_count_returns_int(self, zm_api_live):
        events = zm_api_live.events({"max_events": 5})
        count = events.count()
        assert isinstance(count, int)
        assert count >= 0


@pytest.mark.e2e_readonly
class TestEventAccessors:
    """Verify Event accessor methods return correct types."""

    @pytest.fixture
    def first_event(self, zm_api_live):
        events = zm_api_live.events({"max_events": 5})
        lst = events.list()
        if not lst:
            pytest.skip("No events available on live ZM")
        return lst[0]

    def test_id_is_int(self, first_event):
        assert isinstance(first_event.id(), int)
        assert first_event.id() > 0

    def test_monitor_id_is_int(self, first_event):
        assert isinstance(first_event.monitor_id(), int)
        assert first_event.monitor_id() > 0

    def test_duration_is_float(self, first_event):
        assert isinstance(first_event.duration(), float)
        assert first_event.duration() >= 0

    def test_score_dict(self, first_event):
        score = first_event.score()
        assert isinstance(score, dict)
        assert isinstance(score["total"], float)
        assert isinstance(score["average"], float)
        assert isinstance(score["max"], float)
        assert score["average"] <= score["max"] or score["max"] == 0

    def test_total_frames_is_int(self, first_event):
        assert isinstance(first_event.total_frames(), int)
        assert first_event.total_frames() >= 0

    def test_alarmed_frames_is_int(self, first_event):
        assert isinstance(first_event.alarmed_frames(), int)
        assert first_event.alarmed_frames() >= 0
        assert first_event.alarmed_frames() <= first_event.total_frames()

    def test_name_is_str(self, first_event):
        name = first_event.name()
        # name can be None if the Event Name field is empty
        assert name is None or isinstance(name, str)

    def test_cause_is_str(self, first_event):
        cause = first_event.cause()
        assert cause is None or isinstance(cause, str)


@pytest.mark.e2e_readonly
class TestEventRawGet:
    """Verify Event.get() returns raw dict with expected ZM fields."""

    def test_raw_dict_has_required_fields(self, zm_api_live):
        events = zm_api_live.events({"max_events": 1})
        lst = events.list()
        if not lst:
            pytest.skip("No events available on live ZM")
        raw = lst[0].get()
        assert isinstance(raw, dict)
        # String fields — Name/Cause are always str in ZM's JSON
        for field in ["Name", "Cause"]:
            assert field in raw, f"Missing field: {field}"
            assert isinstance(raw[field], (str, type(None))), \
                f"Expected str or None for raw field {field}, got {type(raw[field])}"
        # Numeric fields — ZM returns int/float, older versions may return str
        for field in ["Id", "MonitorId", "Frames", "AlarmFrames",
                       "TotScore", "AvgScore", "MaxScore"]:
            assert field in raw, f"Missing field: {field}"
            assert isinstance(raw[field], (int, float, str)), \
                f"Expected int/float/str for raw field {field}, got {type(raw[field])}"
        # Length (duration) — numeric
        assert "Length" in raw, "Missing field: Length"
        assert isinstance(raw["Length"], (int, float, str)), \
            f"Expected int/float/str for raw field Length, got {type(raw['Length'])}"


@pytest.mark.e2e_readonly
class TestEventFilters:
    """Verify event filtering works against real ZM."""

    def test_filter_by_max_events(self, zm_api_live):
        events = zm_api_live.events({"max_events": 3})
        assert len(events.list()) <= 3

    def test_filter_by_monitor_id(self, zm_api_live):
        # Get a monitor that exists
        monitors = zm_api_live.monitors()
        if not monitors.list():
            pytest.skip("No monitors on live ZM")
        mid = monitors.list()[0].id()
        events = zm_api_live.events({"mid": mid, "max_events": 5})
        for ev in events.list():
            assert ev.monitor_id() == mid

    def test_filter_by_min_alarmed_frames(self, zm_api_live):
        events = zm_api_live.events({"min_alarmed_frames": 1, "max_events": 5})
        for ev in events.list():
            assert ev.alarmed_frames() >= 1

    def test_empty_result_for_nonexistent_monitor(self, zm_api_live):
        events = zm_api_live.events({"mid": 999999, "max_events": 5})
        assert len(events.list()) == 0


@pytest.mark.e2e_readonly
class TestMonitorEvents:
    """Verify Monitor.events() convenience method."""

    def test_monitor_events_returns_events(self, zm_api_live):
        monitors = zm_api_live.monitors()
        if not monitors.list():
            pytest.skip("No monitors on live ZM")
        mon = monitors.list()[0]
        events = mon.events({"max_events": 3})
        assert isinstance(events, Events)
        for ev in events.list():
            assert ev.monitor_id() == mon.id()


@pytest.mark.e2e_readonly
class TestEventUrls:
    """Verify image/video URL generation."""

    def test_image_url_format(self, zm_api_live):
        events = zm_api_live.events({"max_events": 1})
        lst = events.list()
        if not lst:
            pytest.skip("No events available on live ZM")
        url = lst[0].get_image_url()
        assert isinstance(url, str)
        assert "index.php" in url
        assert "eid=" in url
        # Should contain auth token or credentials
        assert "token=" in url or "auth=" in url

    def test_video_url_format(self, zm_api_live):
        events = zm_api_live.events({"max_events": 1})
        lst = events.list()
        if not lst:
            pytest.skip("No events available on live ZM")
        ev = lst[0]
        url = ev.get_video_url()
        # video_url can be None if no video file
        if ev.video_file():
            assert isinstance(url, str)
            assert "eid=" in url


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_write
class TestEventDelete:
    """Test deleting an event."""

    def test_delete_event(self, zm_api_live, requires_write):
        """Delete the oldest event (least impactful).

        Note: ZM's EventsController.delete() may return flash redirect (None)
        instead of JSON — this test verifies pyzm handles that gracefully.
        """
        events = zm_api_live.events({
            "max_events": 1,
            "sort": "StartTime",
            "direction": "asc",
        })
        lst = events.list()
        if not lst:
            pytest.skip("No events available to delete")
        ev = lst[0]
        eid = ev.id()
        result = ev.delete()
        # ZM may return None (flash redirect) or dict — both acceptable
        assert result is None or isinstance(result, dict)
