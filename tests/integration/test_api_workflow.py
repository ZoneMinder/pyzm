"""Integration test — full login -> monitors -> events -> states workflow."""

import pytest
import responses

from pyzm.api import ZMApi


@pytest.mark.integration
class TestFullWorkflow:

    @responses.activate
    def test_login_monitors_events_states_workflow(
        self,
        zm_options,
        login_success_response,
        monitors_response,
        events_response,
        states_response,
    ):
        """Full end-to-end flow: login -> get monitors -> get events -> get states -> find active."""
        # 1. Login
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )
        api = ZMApi(options=zm_options.copy())
        assert api.authenticated is True

        # 2. Get monitors
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
        )
        monitors = api.monitors()
        assert len(monitors.list()) == 2

        front_door = monitors.find(name="Front Door")
        assert front_door is not None
        assert front_door.id() == 1
        assert front_door.enabled() is True

        backyard = monitors.find(name="Backyard")
        assert backyard is not None
        assert backyard.enabled() is False

        # 3. Get events for a specific monitor
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/events/index/MonitorId =:1.json",
            json=events_response,
            status=200,
        )
        events = api.events(options={"mid": 1})
        assert events.count() == 2
        event_list = events.list()
        assert event_list[0].id() == 100
        assert event_list[0].monitor_id() == 1
        assert event_list[0].cause() == "Motion"

        # 4. Get states
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/states.json",
            json=states_response,
            status=200,
        )
        states = api.states()
        state_list = states.list()
        assert len(state_list) == 3

        # 5. Find active state
        active_states = [s for s in state_list if s.active()]
        assert len(active_states) == 1
        assert active_states[0].name() == "default"

        # Find by name
        away = states.find(name="away")
        assert away is not None
        assert away.active() is False

    @responses.activate
    def test_legacy_auth_workflow(
        self,
        login_legacy_response,
        monitors_response,
    ):
        """Workflow using legacy credentials auth."""
        options = {
            "apiurl": "https://zm.example.com/zm/api",
            "portalurl": "https://zm.example.com/zm",
            "user": "admin",
            "password": "secret",
            "disable_ssl_cert_check": True,
        }

        # Login with legacy credentials
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_legacy_response,
            status=200,
        )
        api = ZMApi(options=options)
        assert api.authenticated is True
        assert api.api_version == "1.0"

        # Get monitors — legacy credentials appended to URL
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
            match_querystring=False,
        )
        monitors = api.monitors()
        assert len(monitors.list()) == 2

        # Verify legacy credentials were in the request URL
        request_url = responses.calls[1].request.url
        assert "auth=abc123hash" in request_url

    @responses.activate
    def test_monitor_operations(
        self,
        zm_options,
        login_success_response,
        monitors_response,
    ):
        """Monitor CRUD operations in a workflow."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )
        api = ZMApi(options=zm_options.copy())

        # Get monitors
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors.json",
            json=monitors_response,
            status=200,
        )
        monitors = api.monitors()
        mon = monitors.find(id=1)

        # Check version info
        version = api.version()
        assert version["status"] == "ok"
        assert version["api_version"] == "2.0"

        # Get monitor status
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/monitors/daemonStatus/id:1/daemon:zmc.json",
            json={"status": True, "statustext": "Running"},
            status=200,
        )
        status = mon.status()
        assert status["status"] is True
