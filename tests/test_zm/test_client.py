"""Tests for pyzm.client -- ZMClient high-level API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import StreamConfig, ZMClientConfig
from pyzm.models.zm import Event, Monitor, Zone


# ===================================================================
# Helpers
# ===================================================================

def _make_mock_api():
    """Create a mock ZMAPI."""
    api = MagicMock()
    api.api_version = "2.0.0"
    api.zm_version = "1.36.12"
    api.api_url = "https://zm.example.com/zm/api"
    api.portal_url = "https://zm.example.com/zm"
    return api


def _sample_monitor_api_data(mid=1, name="Front Door", function="Modect"):
    return {
        "Monitor": {
            "Id": str(mid),
            "Name": name,
            "Function": function,
            "Enabled": "1",
            "Width": "1920",
            "Height": "1080",
            "Type": "Ffmpeg",
        },
        "Monitor_Status": {
            "Status": "Connected",
            "CaptureFPS": "15.0",
            "Capturing": "Capturing",
        },
    }


def _sample_event_api_data(eid=12345, name="Event 12345", monitor_id=1):
    return {
        "Event": {
            "Id": str(eid),
            "Name": name,
            "MonitorId": str(monitor_id),
            "Cause": "Motion",
            "Notes": "",
            "StartTime": "2024-03-15 10:30:00",
            "EndTime": "2024-03-15 10:31:30",
            "Length": "90.5",
            "Frames": "270",
            "AlarmFrames": "45",
            "MaxScore": "97",
            "MaxScoreFrameId": "135",
            "StoragePath": "/var/cache/zoneminder/events",
        }
    }


# ===================================================================
# TestZMClient - Creation
# ===================================================================

class TestZMClientCreation:
    """Tests for ZMClient construction."""

    @patch("pyzm.client.ZMAPI")
    def test_creation_with_url(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api", user="admin", password="secret")

        mock_zmapi_cls.assert_called_once()
        assert client.api is not None

    @patch("pyzm.client.ZMAPI")
    def test_creation_with_config(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        cfg = ZMClientConfig(api_url="https://zm.example.com/zm/api")
        client = ZMClient(config=cfg)

        mock_zmapi_cls.assert_called_once()

    @patch("pyzm.client.ZMAPI")
    def test_creation_auto_appends_api(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm", user="admin", password="secret")

        # The config should have /api appended
        assert client._config.api_url == "https://zm.example.com/zm/api"

    @patch("pyzm.client.ZMAPI")
    def test_creation_auto_appends_api_trailing_slash(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/", user="admin", password="secret")

        assert client._config.api_url == "https://zm.example.com/zm/api"

    @patch("pyzm.client.ZMAPI")
    def test_creation_does_not_double_append_api(self, mock_zmapi_cls):
        mock_zmapi_cls.return_value = _make_mock_api()

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api", user="admin", password="secret")

        assert client._config.api_url == "https://zm.example.com/zm/api"

    @patch("pyzm.client.ZMAPI")
    def test_creation_no_url_no_config_raises(self, mock_zmapi_cls):
        from pyzm.client import ZMClient
        with pytest.raises(ValueError, match="Either 'url' or 'config'"):
            ZMClient()

    @patch("pyzm.client.ZMAPI")
    def test_properties(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        assert client.zm_version == "1.36.12"
        assert client.api_version == "2.0.0"


# ===================================================================
# TestZMClient - Monitors
# ===================================================================

class TestZMClientMonitors:
    """Tests for monitors-related methods."""

    @patch("pyzm.client.ZMAPI")
    def test_monitors_fetches_and_caches(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(2, "Back Yard"),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        monitors = client.monitors()
        assert len(monitors) == 2
        assert monitors[0].name == "Front Door"
        assert monitors[1].name == "Back Yard"

        # Second call should use cache (API not called again)
        mock_api.get.reset_mock()
        monitors2 = client.monitors()
        mock_api.get.assert_not_called()
        assert monitors2 is monitors

    @patch("pyzm.client.ZMAPI")
    def test_monitors_force_reload(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [_sample_monitor_api_data(1, "Front Door")]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.monitors()  # first fetch
        mock_api.get.reset_mock()

        # Force reload should call API again
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(3, "Side Gate"),
            ]
        }
        monitors = client.monitors(force_reload=True)
        mock_api.get.assert_called_once()
        assert len(monitors) == 2

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_id_from_cache(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "monitors": [
                _sample_monitor_api_data(1, "Front Door"),
                _sample_monitor_api_data(2, "Back Yard"),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        m = client.monitor(2)
        assert m.id == 2
        assert m.name == "Back Yard"

    @patch("pyzm.client.ZMAPI")
    def test_monitor_by_id_not_found_fallback(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        # monitors() returns only monitor 1
        mock_api.get.side_effect = [
            {"monitors": [_sample_monitor_api_data(1, "Front Door")]},
            {"monitor": _sample_monitor_api_data(99, "Unknown")},
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        m = client.monitor(99)
        assert m.id == 99

    @patch("pyzm.client.ZMAPI")
    def test_monitor_not_found_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.side_effect = [
            {"monitors": []},
            {},  # empty response for direct lookup
        ]
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found"):
            client.monitor(999)


# ===================================================================
# TestZMClient - Events
# ===================================================================

class TestZMClientEvents:
    """Tests for events-related methods."""

    @patch("pyzm.client.ZMAPI")
    def test_events_basic(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "events": [
                _sample_event_api_data(100),
                _sample_event_api_data(101),
            ]
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        events = client.events()
        assert len(events) == 2
        assert events[0].id == 100
        assert events[1].id == 101

    @patch("pyzm.client.ZMAPI")
    def test_events_filter_by_monitor(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"events": [_sample_event_api_data(100)]}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.events(monitor_id=2)

        # Verify the URL uses CakePHP path filter syntax
        call_url = mock_api.get.call_args[0][0]
        assert "MonitorId:2" in call_url

    @patch("pyzm.client.ZMAPI")
    def test_events_filter_building(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {"events": []}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.events(
            monitor_id=1,
            min_alarm_frames=5,
            object_only=True,
            limit=50,
        )

        call_url = mock_api.get.call_args[0][0]
        assert "MonitorId:1" in call_url
        assert "AlarmFrames >=:5" in call_url
        assert "Notes REGEXP:detected" in call_url
        call_params = mock_api.get.call_args[1].get("params", {})
        assert call_params["limit"] == "50"

    @patch("pyzm.client.ZMAPI")
    def test_event_single_fetch(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {
            "event": _sample_event_api_data(42)
        }
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        ev = client.event(42)
        assert ev.id == 42

    @patch("pyzm.client.ZMAPI")
    def test_event_not_found_raises(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_api.get.return_value = {}
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        with pytest.raises(ValueError, match="not found"):
            client.event(99999)


# ===================================================================
# TestZMClient - Frames
# ===================================================================

class TestZMClientFrames:
    """Tests for get_event_frames."""

    @patch("pyzm.client.FrameExtractor")
    @patch("pyzm.client.ZMAPI")
    def test_get_event_frames(self, mock_zmapi_cls, mock_extractor_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.models.zm import Frame as ZMFrame
        mock_frame = ZMFrame(frame_id=1, event_id=12345)
        mock_image = MagicMock()
        mock_image.shape = (480, 640, 3)

        mock_extractor = MagicMock()
        mock_extractor.extract_frames.return_value = [(mock_frame, mock_image)]
        mock_extractor.original_shape = (480, 640)
        mock_extractor_cls.return_value = mock_extractor

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        frames, image_dims = client.get_event_frames(12345)
        assert len(frames) == 1
        assert frames[0][0] == 1  # frame_id
        assert frames[0][1] is mock_image
        assert image_dims["original"] == (480, 640)
        assert image_dims["resized"] is None  # no resize happened

    @patch("pyzm.client.FrameExtractor")
    @patch("pyzm.client.ZMAPI")
    def test_get_event_frames_with_config(self, mock_zmapi_cls, mock_extractor_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        mock_extractor = MagicMock()
        mock_extractor.extract_frames.return_value = []
        mock_extractor_cls.return_value = mock_extractor

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        sc = StreamConfig(frame_set=["1", "5"], max_frames=2)
        client.get_event_frames(12345, stream_config=sc)

        # Verify FrameExtractor was created with the correct config
        mock_extractor_cls.assert_called_once()
        call_kwargs = mock_extractor_cls.call_args
        assert call_kwargs[1]["stream_config"] is sc or call_kwargs[0][1] is sc


# ===================================================================
# TestZMClient - update_event_notes
# ===================================================================

class TestZMClientUpdateNotes:
    """Tests for update_event_notes."""

    @patch("pyzm.client.ZMAPI")
    def test_update_event_notes(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.update_event_notes(12345, "person:97% detected")

        mock_api.put.assert_called_once_with(
            "events/12345.json",
            data={"Event[Notes]": "person:97% detected"},
        )


# ===================================================================
# TestZMClient - tag_event
# ===================================================================

class TestZMClientTagEvent:
    """Tests for tag_event (direct DB implementation)."""

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_tag_event_creates_new_tag(self, mock_get_db, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # tag doesn't exist
        mock_cursor.lastrowid = 7
        mock_get_db.return_value = mock_conn

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")
        client.tag_event(12345, ["person"])

        # Should INSERT new tag then link it
        calls = mock_cursor.execute.call_args_list
        assert any("SELECT Id FROM Tags" in str(c) for c in calls)
        assert any("INSERT INTO Tags" in str(c) for c in calls)
        assert any("INSERT INTO Events_Tags" in str(c) for c in calls)
        mock_conn.commit.assert_called()

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_tag_event_links_existing_tag(self, mock_get_db, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {"Id": 42}  # tag exists
        mock_get_db.return_value = mock_conn

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")
        client.tag_event(12345, ["person"])

        calls = mock_cursor.execute.call_args_list
        # Should UPDATE existing tag's LastAssignedDate, then link
        assert any("UPDATE Tags SET LastAssignedDate" in str(c) for c in calls)
        assert any("INSERT INTO Events_Tags" in str(c) for c in calls)
        # Should NOT insert a new tag
        assert not any("INSERT INTO Tags" in str(c) and "Name" in str(c) for c in calls)
        mock_conn.commit.assert_called()

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_tag_event_deduplicates_labels(self, mock_get_db, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_cursor.lastrowid = 1
        mock_get_db.return_value = mock_conn

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")
        client.tag_event(12345, ["person", "person", "car", "car"])

        # Should only process 2 unique labels, not 4
        select_calls = [c for c in mock_cursor.execute.call_args_list
                        if "SELECT Id FROM Tags" in str(c)]
        assert len(select_calls) == 2

    @patch("pyzm.client.ZMAPI")
    def test_tag_event_empty_labels(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")
        client.tag_event(12345, [])
        # No DB calls should happen

    @patch("pyzm.client.ZMAPI")
    @patch("pyzm.zm.db.get_zm_db")
    def test_tag_event_no_db_graceful(self, mock_get_db, mock_zmapi_cls):
        """When DB is unavailable, tag_event logs warning and returns."""
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api
        mock_get_db.return_value = None

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")
        client.tag_event(12345, ["person"])  # should not raise


# ===================================================================
# TestZMClient - State Management
# ===================================================================

class TestZMClientState:
    """Tests for state management."""

    @patch("pyzm.client.ZMAPI")
    def test_set_state(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.set_state("restart")
        mock_api.get.assert_called_with("states/change/restart.json")

    @patch("pyzm.client.ZMAPI")
    def test_start(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.start()
        mock_api.get.assert_called_with("states/change/start.json")

    @patch("pyzm.client.ZMAPI")
    def test_stop(self, mock_zmapi_cls):
        mock_api = _make_mock_api()
        mock_zmapi_cls.return_value = mock_api

        from pyzm.client import ZMClient
        client = ZMClient(url="https://zm.example.com/zm/api")

        client.stop()
        mock_api.get.assert_called_with("states/change/stop.json")
