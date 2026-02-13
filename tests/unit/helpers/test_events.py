"""Edge-case tests for Events helper — URL filter building logic."""

from unittest.mock import MagicMock, patch

import pytest
import responses

from pyzm.helpers.Events import Events


def _make_api_mock(events_data=None):
    """Create a mock API that returns canned event data."""
    if events_data is None:
        events_data = {
            "events": [],
            "pagination": {"count": 0, "current": 0, "nextPage": False},
        }
    api = MagicMock()
    api.api_url = "https://zm.example.com/zm/api"
    api._make_request.return_value = events_data
    return api


@pytest.mark.unit
class TestEventFilters:

    def test_filter_url_with_monitor_id(self):
        """MonitorId filter appears in the URL."""
        api = _make_api_mock()

        Events(api=api, options={"mid": 5})

        call_url = api._make_request.call_args[1]["url"]
        assert "/MonitorId =:5" in call_url

    def test_filter_url_with_object_only(self):
        """object_only appends REGEXP filter."""
        api = _make_api_mock()

        Events(api=api, options={"object_only": True})

        call_url = api._make_request.call_args[1]["url"]
        assert "/Notes REGEXP:detected:" in call_url

    def test_filter_url_with_event_id(self):
        """event_id filter appears in URL."""
        api = _make_api_mock()

        Events(api=api, options={"event_id": "42"})

        call_url = api._make_request.call_args[1]["url"]
        assert "/Id=:42" in call_url

    def test_filter_url_with_alarmed_frames(self):
        """min/max alarmed frames filters appear in URL."""
        api = _make_api_mock()

        Events(
            api=api,
            options={"min_alarmed_frames": 10, "max_alarmed_frames": 100},
        )

        call_url = api._make_request.call_args[1]["url"]
        assert "/AlarmFrames >=:10" in call_url
        assert "/AlarmFrames <=:100" in call_url

    @patch("pyzm.helpers.Events.dateparser")
    def test_filter_url_with_time_range(self, mock_dateparser):
        """'from' with 'X to Y' splits into start/end time filters."""
        from datetime import datetime

        start = datetime(2024, 1, 15, 9, 0, 0)
        end = datetime(2024, 1, 15, 10, 0, 0)
        mock_dateparser.parse.side_effect = [start, end]

        api = _make_api_mock()

        Events(api=api, options={"from": "1 hour ago to now"})

        call_url = api._make_request.call_args[1]["url"]
        assert "/StartTime >=:2024-01-15 09:00:00" in call_url
        assert "/StartTime <=:2024-01-15 10:00:00" in call_url


@pytest.mark.unit
class TestPagination:

    def test_pagination_stops_at_max_events(self):
        """Loop terminates when currevents >= max_events."""
        page1 = {
            "events": [
                {"Event": {"Id": "1", "Name": "E1", "MonitorId": "1",
                           "Cause": "", "Notes": "", "StartTime": "",
                           "EndTime": "", "Length": "1", "Frames": "1",
                           "AlarmFrames": "1", "TotScore": "1",
                           "AvgScore": "1", "MaxScore": "1",
                           "DefaultVideo": None, "FileSystemPath": ""}},
            ],
            "pagination": {
                "count": 5,
                "current": 1,
                "nextPage": True,
                "page": 1,
            },
        }
        page2 = {
            "events": [
                {"Event": {"Id": "2", "Name": "E2", "MonitorId": "1",
                           "Cause": "", "Notes": "", "StartTime": "",
                           "EndTime": "", "Length": "1", "Frames": "1",
                           "AlarmFrames": "1", "TotScore": "1",
                           "AvgScore": "1", "MaxScore": "1",
                           "DefaultVideo": None, "FileSystemPath": ""}},
            ],
            "pagination": {
                "count": 5,
                "current": 1,
                "nextPage": True,
                "page": 2,
            },
        }

        api = MagicMock()
        api.api_url = "https://zm.example.com/zm/api"
        api._make_request.side_effect = [page1, page2]

        events = Events(api=api, options={"max_events": 2})

        # Should have fetched 2 pages (1 event each, limit=2)
        assert len(events.list()) == 2
        assert api._make_request.call_count == 2
