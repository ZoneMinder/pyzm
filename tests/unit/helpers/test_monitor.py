"""Edge-case tests for Monitor helper — only logic NOT covered by API tests."""

from unittest.mock import MagicMock

import pytest

from pyzm.helpers.Monitor import Monitor


def _make_monitor(overrides=None):
    """Create a Monitor instance with a mock API and test data."""
    data = {
        "Monitor": {
            "Id": "5",
            "Name": "Test Camera",
            "Function": "Modect",
            "Enabled": "1",
            "Type": "Ffmpeg",
            "Width": "1920",
            "Height": "1080",
        }
    }
    if overrides:
        data["Monitor"].update(overrides)
    api = MagicMock()
    api.api_url = "https://zm.example.com/zm/api"
    return Monitor(monitor=data, api=api)


@pytest.mark.unit
class TestMonitorEnabled:

    def test_monitor_enabled_false(self):
        """Enabled == '0' returns False."""
        mon = _make_monitor({"Enabled": "0"})
        assert mon.enabled() is False

    def test_monitor_enabled_true(self):
        """Enabled == '1' returns True."""
        mon = _make_monitor({"Enabled": "1"})
        assert mon.enabled() is True


@pytest.mark.unit
class TestSetParameter:

    def test_set_parameter_builds_payload(self):
        """Correct Monitor[Function] etc. payload keys are built."""
        mon = _make_monitor()

        mon.set_parameter(options={
            "function": "Record",
            "name": "New Name",
            "enabled": True,
        })

        mon.api._make_request.assert_called_once()
        call_kwargs = mon.api._make_request.call_args
        payload = call_kwargs[1]["payload"]
        assert payload["Monitor[Function]"] == "Record"
        assert payload["Monitor[Name]"] == "New Name"
        assert payload["Monitor[Enabled]"] == "1"

    def test_set_parameter_disabled(self):
        """enabled=False sends '0'."""
        mon = _make_monitor()

        mon.set_parameter(options={"enabled": False})

        call_kwargs = mon.api._make_request.call_args
        payload = call_kwargs[1]["payload"]
        assert payload["Monitor[Enabled]"] == "0"

    def test_set_parameter_with_raw(self):
        """Raw parameter passthrough works."""
        mon = _make_monitor()

        mon.set_parameter(options={
            "raw": {"Monitor[Colours]": "4", "Monitor[Method]": "simple"},
        })

        call_kwargs = mon.api._make_request.call_args
        payload = call_kwargs[1]["payload"]
        assert payload["Monitor[Colours]"] == "4"
        assert payload["Monitor[Method]"] == "simple"

    def test_set_parameter_empty_noop(self):
        """No payload = no API call."""
        mon = _make_monitor()

        result = mon.set_parameter(options={})

        mon.api._make_request.assert_not_called()
        assert result is None


@pytest.mark.unit
class TestArmDisarm:

    def test_arm_url(self):
        """arm() calls correct alarm command URL."""
        mon = _make_monitor()

        mon.arm()

        call_kwargs = mon.api._make_request.call_args
        url = call_kwargs[1]["url"]
        assert "/monitors/alarm/id:5/command:on.json" in url

    def test_disarm_url(self):
        """disarm() calls correct alarm command URL."""
        mon = _make_monitor()

        mon.disarm()

        call_kwargs = mon.api._make_request.call_args
        url = call_kwargs[1]["url"]
        assert "/monitors/alarm/id:5/command:off.json" in url
