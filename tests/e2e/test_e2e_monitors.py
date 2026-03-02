"""E2E tests for Monitors and Monitor objects against a live ZoneMinder.

Readonly tests verify list/find/accessors. Write tests create, modify,
and delete monitors (with auto-cleanup).
"""

import pytest

from pyzm.helpers.Monitors import Monitors
from pyzm.helpers.Monitor import Monitor

# ---------------------------------------------------------------------------
# Readonly tests
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.e2e]


@pytest.mark.e2e_readonly
class TestMonitorList:
    """Verify monitors() returns a valid Monitors collection."""

    def test_returns_monitors_instance(self, zm_api_live):
        result = zm_api_live.monitors()
        assert isinstance(result, Monitors)

    def test_list_returns_list(self, zm_api_live):
        monitors = zm_api_live.monitors()
        lst = monitors.list()
        assert isinstance(lst, list)

    def test_list_nonempty(self, zm_api_live):
        monitors = zm_api_live.monitors()
        assert len(monitors.list()) > 0, "Expected at least one monitor on live ZM"

    def test_items_are_monitor_objects(self, zm_api_live):
        monitors = zm_api_live.monitors()
        for mon in monitors.list():
            assert isinstance(mon, Monitor)


@pytest.mark.e2e_readonly
class TestMonitorAccessors:
    """Verify Monitor accessor methods return correct types."""

    def test_id_is_int(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        assert isinstance(mon.id(), int)
        assert mon.id() > 0

    def test_name_is_str(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        assert isinstance(mon.name(), str)
        assert len(mon.name()) > 0

    def test_function_is_valid(self, zm_api_live):
        valid_functions = {
            "None", "Monitor", "Modect", "Record",
            "Mocord", "Nodect",
        }
        mon = zm_api_live.monitors().list()[0]
        func = mon.function()
        assert isinstance(func, str)
        assert func in valid_functions, f"Unexpected function: {func}"

    def test_enabled_is_bool(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        assert isinstance(mon.enabled(), bool)

    def test_dimensions_dict(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        dims = mon.dimensions()
        assert isinstance(dims, dict)
        assert isinstance(dims["width"], int)
        assert isinstance(dims["height"], int)
        assert dims["width"] > 0
        assert dims["height"] > 0

    def test_type_is_str(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        assert isinstance(mon.type(), str)


@pytest.mark.e2e_readonly
class TestMonitorRawGet:
    """Verify Monitor.get() returns raw dict with expected ZM fields."""

    def test_raw_dict_has_required_fields(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        raw = mon.get()
        assert isinstance(raw, dict)
        # String fields — always str in ZM's JSON
        for field in ["Name", "Function", "Type"]:
            assert field in raw, f"Missing field: {field}"
            assert isinstance(raw[field], str), \
                f"Expected str for raw field {field}, got {type(raw[field])}"
        # Id — confirmed int on real ZM
        assert isinstance(raw["Id"], int), \
            f"Expected int for raw field Id, got {type(raw['Id'])}"
        # Numeric fields — ZM returns int, older versions may return str
        for field in ["Enabled", "Width", "Height"]:
            assert field in raw, f"Missing field: {field}"
            assert isinstance(raw[field], (int, str)), \
                f"Expected int or str for raw field {field}, got {type(raw[field])}"


@pytest.mark.e2e_readonly
class TestMonitorFind:
    """Verify Monitors.find() by id and name."""

    def test_find_by_id(self, zm_api_live):
        monitors = zm_api_live.monitors()
        first = monitors.list()[0]
        found = monitors.find(id=first.id())
        assert found is not None
        assert found.id() == first.id()

    def test_find_by_name(self, zm_api_live):
        monitors = zm_api_live.monitors()
        first = monitors.list()[0]
        found = monitors.find(name=first.name())
        assert found is not None
        assert found.name() == first.name()

    def test_find_nonexistent_returns_none(self, zm_api_live):
        monitors = zm_api_live.monitors()
        found = monitors.find(name="pyzm_e2e_nonexistent_monitor_xyz")
        assert found is None


@pytest.mark.e2e_readonly
class TestMonitorDaemonStatus:
    """Verify daemon status returns a response."""

    def test_status_returns_dict(self, zm_api_live):
        mon = zm_api_live.monitors().list()[0]
        result = mon.status()
        # status() may return a dict or None depending on ZM configuration
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_write
class TestMonitorAdd:
    """Test creating a new monitor."""

    def test_add_monitor(self, zm_api_live, e2e_monitor_factory):
        mon, result = e2e_monitor_factory(name="add_test", function="Monitor")
        assert mon is not None, "Failed to find newly created monitor"
        assert mon.name() == "pyzm_e2e_test_add_test"
        assert mon.function() == "Monitor"

    def test_add_monitor_with_dimensions(self, zm_api_live, e2e_monitor_factory):
        mon, result = e2e_monitor_factory(
            name="dims_test",
            function="Monitor",
            width=320,
            height=240,
        )
        assert mon is not None
        dims = mon.dimensions()
        assert dims["width"] == 320
        assert dims["height"] == 240


@pytest.mark.e2e_write
class TestMonitorSetParameter:
    """Test modifying monitor parameters."""

    def test_set_function(self, zm_api_live, e2e_monitor_factory):
        mon, _ = e2e_monitor_factory(name="setparam_func", function="Monitor")
        assert mon is not None
        mon.set_parameter({"function": "Modect"})
        # Reload to verify
        monitors = zm_api_live.monitors({"force_reload": True})
        updated = monitors.find(name="pyzm_e2e_test_setparam_func")
        assert updated is not None
        assert updated.function() == "Modect"

    def test_set_name(self, zm_api_live, e2e_monitor_factory):
        mon, _ = e2e_monitor_factory(name="setparam_name", function="Monitor")
        assert mon is not None
        mon.set_parameter({"name": "pyzm_e2e_test_setparam_renamed"})
        monitors = zm_api_live.monitors({"force_reload": True})
        updated = monitors.find(name="pyzm_e2e_test_setparam_renamed")
        assert updated is not None

    def test_set_enabled(self, zm_api_live, e2e_monitor_factory):
        mon, _ = e2e_monitor_factory(name="setparam_enabled", function="Monitor", enabled=False)
        assert mon is not None
        assert mon.enabled() is False
        mon.set_parameter({"enabled": True})
        monitors = zm_api_live.monitors({"force_reload": True})
        updated = monitors.find(name="pyzm_e2e_test_setparam_enabled")
        assert updated is not None
        assert updated.enabled() is True

    def test_set_raw_parameter(self, zm_api_live, e2e_monitor_factory):
        mon, _ = e2e_monitor_factory(name="setparam_raw", function="Monitor")
        assert mon is not None
        mon.set_parameter({"raw": {"Monitor[Colours]": "4"}})
        monitors = zm_api_live.monitors({"force_reload": True})
        updated = monitors.find(name="pyzm_e2e_test_setparam_raw")
        assert updated is not None
        assert str(updated.get()["Colours"]) == "4"


@pytest.mark.e2e_write
class TestMonitorDelete:
    """Test deleting a monitor."""

    def test_delete_monitor(self, zm_api_live, e2e_monitor_factory):
        mon, _ = e2e_monitor_factory(name="delete_test", function="Monitor")
        assert mon is not None
        mid = mon.id()
        # Delete it manually (factory will also try in teardown, which is fine)
        result = mon.delete()
        # ZM may return None (flash redirect) or a dict — both are acceptable
        assert result is None or isinstance(result, dict)
        # Verify it's gone
        monitors = zm_api_live.monitors({"force_reload": True})
        assert monitors.find(id=mid) is None


@pytest.mark.e2e_write
class TestMonitorAddFlashHandling:
    """Test that add() handles flash() responses from ZM."""

    def test_add_returns_response(self, zm_api_live, e2e_monitor_factory):
        """ZM's MonitorsController.add() may return flash redirect or JSON."""
        _, result = e2e_monitor_factory(name="flash_test", function="Monitor")
        # Result may be None (flash), dict (JSON), or response object
        # The important thing is it doesn't crash
        assert result is None or isinstance(result, dict)
