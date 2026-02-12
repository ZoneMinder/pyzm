"""Tests for ZMApi public methods (monitors, events, states, configs, etc.).

These are the primary coverage layer -- helper objects get tested as part
of the API contract.  If an API test already asserts a helper behavior,
there is no separate helper test for it.
"""

import pytest
import responses

from pyzm.helpers.Configs import Configs
from pyzm.helpers.Event import Event
from pyzm.helpers.Events import Events
from pyzm.helpers.Monitor import Monitor
from pyzm.helpers.Monitors import Monitors
from pyzm.helpers.State import State
from pyzm.helpers.States import States


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMonitors:

    @responses.activate
    def test_monitors_returns_monitors_object(
        self, zm_api, monitors_response
    ):
        """monitors() returns a Monitors instance with correct Monitor objects."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
        )

        result = zm_api.monitors()

        assert isinstance(result, Monitors)
        monitors = result.list()
        assert len(monitors) == 2

        m1 = monitors[0]
        assert isinstance(m1, Monitor)
        assert m1.id() == 1
        assert m1.name() == "Front Door"
        assert m1.function() == "Modect"
        assert m1.enabled() is True
        assert m1.type() == "Ffmpeg"
        assert m1.dimensions() == {"width": 1920, "height": 1080}

        m2 = monitors[1]
        assert m2.id() == 2
        assert m2.name() == "Backyard"
        assert m2.enabled() is False

    @responses.activate
    def test_monitors_cached(self, zm_api, monitors_response):
        """Second call returns cached Monitors without new API request."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
        )

        first = zm_api.monitors()
        second = zm_api.monitors()

        assert first is second
        assert len(responses.calls) == 1

    @responses.activate
    def test_monitors_force_reload(self, zm_api, monitors_response):
        """force_reload=True re-fetches from API."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
        )

        zm_api.monitors()
        zm_api.monitors(options={"force_reload": True})

        assert len(responses.calls) == 2


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvents:

    @responses.activate
    def test_events_returns_events_object(self, zm_api, events_response):
        """events() returns an Events instance with correct Event objects."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/events/index.json",
            json=events_response,
            status=200,
        )

        result = zm_api.events()

        assert isinstance(result, Events)
        events = result.list()
        assert len(events) == 2
        assert result.count() == 2

        e1 = events[0]
        assert isinstance(e1, Event)
        assert e1.id() == 100
        assert e1.name() == "Event 100"
        assert e1.monitor_id() == 1
        assert e1.cause() == "Motion"
        assert e1.notes() == "detected:person"
        assert e1.duration() == 60.5
        assert e1.total_frames() == 150
        assert e1.alarmed_frames() == 45
        assert e1.score() == {"total": 500.0, "average": 11.1, "max": 85.0}
        assert e1.video_file() == "100-video.mp4"
        assert e1.fspath() == "/var/lib/zoneminder/events/1/2024-01-15/100"

        e2 = events[1]
        assert e2.id() == 99
        assert e2.video_file() is None


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStates:

    @responses.activate
    def test_states_returns_states_object(self, zm_api, states_response):
        """states() returns a States instance with correct State objects."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/states.json",
            json=states_response,
            status=200,
        )

        result = zm_api.states()

        assert isinstance(result, States)
        states = result.list()
        assert len(states) == 3

        s1 = states[0]
        assert isinstance(s1, State)
        assert s1.id() == 1
        assert s1.name() == "default"
        assert s1.active() is True
        assert s1.definition() == "ZM default state"

        s2 = states[1]
        assert s2.id() == 2
        assert s2.name() == "away"
        assert s2.active() is False


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigs:

    @responses.activate
    def test_configs_returns_configs_object(self, zm_api, configs_response):
        """configs() returns a Configs instance."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/configs.json",
            json=configs_response,
            status=200,
        )

        result = zm_api.configs()

        assert isinstance(result, Configs)
        assert len(result.list()) == 3

    @responses.activate
    def test_configs_cached(self, zm_api, configs_response):
        """Second call returns cached Configs."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/configs.json",
            json=configs_response,
            status=200,
        )

        first = zm_api.configs()
        second = zm_api.configs()

        assert first is second
        assert len(responses.calls) == 1

    @responses.activate
    def test_configs_find_no_args_returns_none(self, zm_api, configs_response):
        """find() with no args returns None without iterating."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/configs.json",
            json=configs_response,
            status=200,
        )
        configs = zm_api.configs()
        assert configs.find() is None

    @responses.activate
    def test_configs_set_name_none_returns_none(self, zm_api, configs_response):
        """set(name=None) returns None (early guard)."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/configs.json",
            json=configs_response,
            status=200,
        )
        configs = zm_api.configs()
        assert configs.set(name=None, val="anything") is None

    @responses.activate
    def test_configs_set_val_none_returns_none(self, zm_api, configs_response):
        """set(val=None) returns None (early guard)."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/configs.json",
            json=configs_response,
            status=200,
        )
        configs = zm_api.configs()
        assert configs.set(name="ZM_AUTH_TYPE", val=None) is None


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVersion:

    def test_version_returns_dict(self, zm_api):
        """version() returns status, api_version, zm_version."""
        result = zm_api.version()

        assert result["status"] == "ok"
        assert result["api_version"] == "2.0"
        assert result["zm_version"] == "1.36.32"

    @responses.activate
    def test_version_unauthenticated(self, zm_options_no_auth, version_response):
        """version() returns error when not authenticated."""
        from pyzm.api import ZMApi

        # The no-auth login path sets authenticated=True too, so we need
        # to force the unauthenticated state by making login fail
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/getVersion.json",
            json=version_response,
            status=200,
        )
        api = ZMApi(options=zm_options_no_auth.copy())
        # Force unauthenticated state
        api.authenticated = False

        result = api.version()
        assert result["status"] == "error"
        assert "reason" in result


# ---------------------------------------------------------------------------
# set_state / restart / stop / start
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStateControl:

    @responses.activate
    def test_set_state(self, zm_api):
        """set_state calls correct URL."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/states/change/mystate.json",
            json={"result": "ok"},
            status=200,
        )

        result = zm_api.set_state("mystate")

        assert result == {"result": "ok"}
        assert "states/change/mystate.json" in responses.calls[0].request.url

    @responses.activate
    def test_restart_calls_set_state(self, zm_api):
        """restart() calls set_state('restart')."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/states/change/restart.json",
            json={"result": "ok"},
            status=200,
        )

        zm_api.restart()

        assert "states/change/restart.json" in responses.calls[0].request.url

    @responses.activate
    def test_stop_calls_set_state(self, zm_api):
        """stop() calls set_state('stop')."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/states/change/stop.json",
            json={"result": "ok"},
            status=200,
        )

        zm_api.stop()

        assert "states/change/stop.json" in responses.calls[0].request.url

    @responses.activate
    def test_start_calls_set_state(self, zm_api):
        """start() calls set_state('start')."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/states/change/start.json",
            json={"result": "ok"},
            status=200,
        )

        zm_api.start()

        assert "states/change/start.json" in responses.calls[0].request.url

    def test_set_state_none_returns_none(self, zm_api):
        """set_state(None) returns None without making a request."""
        result = zm_api.set_state(None)
        assert result is None


# ---------------------------------------------------------------------------
# get_auth
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetAuth:

    def test_get_auth_jwt(self, zm_api):
        """get_auth returns 'token=XXX' for JWT auth."""
        result = zm_api.get_auth()
        assert result == "token=test_access_token_abc123"

    def test_get_auth_legacy(self, zm_api_legacy):
        """get_auth returns legacy credentials string."""
        result = zm_api_legacy.get_auth()
        assert result == "auth=abc123hash"

    @responses.activate
    def test_get_auth_disabled(self, zm_options_no_auth, version_response):
        """get_auth returns empty string when auth disabled."""
        from pyzm.api import ZMApi

        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/getVersion.json",
            json=version_response,
            status=200,
        )

        api = ZMApi(options=zm_options_no_auth.copy())
        assert api.get_auth() == ""
