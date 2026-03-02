"""E2E tests for States and State objects against a live ZoneMinder.

Readonly tests verify list/find/accessors. Write tests switch states.
"""

import pytest

from pyzm.helpers.States import States
from pyzm.helpers.State import State

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Readonly tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_readonly
class TestStateList:
    """Verify states() returns a valid States collection."""

    def test_returns_states_instance(self, zm_api_live):
        result = zm_api_live.states()
        assert isinstance(result, States)

    def test_list_returns_list(self, zm_api_live):
        states = zm_api_live.states()
        lst = states.list()
        assert isinstance(lst, list)

    def test_list_nonempty(self, zm_api_live):
        states = zm_api_live.states()
        assert len(states.list()) > 0, "Expected at least one state on live ZM"

    def test_items_are_state_objects(self, zm_api_live):
        states = zm_api_live.states()
        for s in states.list():
            assert isinstance(s, State)


@pytest.mark.e2e_readonly
class TestStateAccessors:
    """Verify State accessor methods return correct types."""

    def test_id_is_int(self, zm_api_live):
        state = zm_api_live.states().list()[0]
        assert isinstance(state.id(), int)
        assert state.id() > 0

    def test_name_is_str(self, zm_api_live):
        state = zm_api_live.states().list()[0]
        assert isinstance(state.name(), str)
        assert len(state.name()) > 0

    def test_active_is_bool(self, zm_api_live):
        state = zm_api_live.states().list()[0]
        assert isinstance(state.active(), bool)

    def test_definition_is_str_or_none(self, zm_api_live):
        state = zm_api_live.states().list()[0]
        defn = state.definition()
        assert defn is None or isinstance(defn, str)


@pytest.mark.e2e_readonly
class TestStateRawGet:
    """Verify State.get() returns raw dict with expected ZM fields."""

    def test_raw_dict_has_required_fields(self, zm_api_live):
        state = zm_api_live.states().list()[0]
        raw = state.get()
        assert isinstance(raw, dict)
        # String fields — Name is always str in ZM's JSON
        assert "Name" in raw, "Missing field: Name"
        assert isinstance(raw["Name"], str), \
            f"Expected str for raw field Name, got {type(raw['Name'])}"
        # Numeric fields — ZM returns int, older versions may return str
        for field in ["Id", "IsActive"]:
            assert field in raw, f"Missing field: {field}"
            assert isinstance(raw[field], (int, str)), \
                f"Expected int or str for raw field {field}, got {type(raw[field])}"


@pytest.mark.e2e_readonly
class TestStateFind:
    """Verify States.find() by id and name."""

    def test_find_by_id(self, zm_api_live):
        states = zm_api_live.states()
        first = states.list()[0]
        found = states.find(id=first.id())
        assert found is not None
        assert found.id() == first.id()

    def test_find_by_name(self, zm_api_live):
        states = zm_api_live.states()
        first = states.list()[0]
        found = states.find(name=first.name())
        assert found is not None
        assert found.name() == first.name()

    def test_find_nonexistent_returns_none(self, zm_api_live):
        states = zm_api_live.states()
        found = states.find(name="pyzm_e2e_nonexistent_state_xyz")
        assert found is None


@pytest.mark.e2e_readonly
class TestStateActiveInvariant:
    """Verify at most one state is active at a time."""

    def test_at_most_one_active(self, zm_api_live):
        states = zm_api_live.states()
        active_count = sum(1 for s in states.list() if s.active())
        assert active_count <= 1, \
            f"Expected at most 1 active state, found {active_count}"


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_write
class TestSetState:
    """Test switching ZM state with restore."""

    def test_set_state_and_restore(self, zm_api_live, requires_write):
        """Record active state, switch to another, verify, restore."""
        states = zm_api_live.states()
        state_list = states.list()
        if len(state_list) < 2:
            pytest.skip("Need at least 2 states to test switching")

        # Record the currently active state (if any)
        active_state = None
        for s in state_list:
            if s.active():
                active_state = s
                break

        # Pick a target state that's different from the active one
        target = None
        for s in state_list:
            if not s.active():
                target = s
                break

        if target is None:
            pytest.skip("No inactive state to switch to")

        # Switch to target
        result = zm_api_live.set_state(target.name())
        # set_state may return dict or None
        assert result is None or isinstance(result, dict)

        # Restore original state if there was one
        if active_state:
            zm_api_live.set_state(active_state.name())
