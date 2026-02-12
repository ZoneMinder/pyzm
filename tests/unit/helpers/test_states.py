"""Edge-case tests for States helper — search/find logic."""

from unittest.mock import MagicMock

import pytest

from pyzm.helpers.State import State
from pyzm.helpers.States import States


def _make_states():
    """Build a States-like object with pre-populated state list (no API call)."""
    states_obj = States.__new__(States)
    states_obj.api = MagicMock()
    states_obj.states = [
        State(
            state={"State": {"Id": "1", "Name": "default", "IsActive": "1", "Definition": ""}},
            api=states_obj.api,
        ),
        State(
            state={"State": {"Id": "2", "Name": "Away", "IsActive": "0", "Definition": "All active"}},
            api=states_obj.api,
        ),
        State(
            state={"State": {"Id": "3", "Name": "Home", "IsActive": "0", "Definition": ""}},
            api=states_obj.api,
        ),
    ]
    return states_obj


@pytest.mark.unit
class TestStatesFind:

    def test_find_by_id(self):
        """Exact id match returns correct State."""
        states = _make_states()

        result = states.find(id=2)

        assert result is not None
        assert result.id() == 2
        assert result.name() == "Away"

    def test_find_by_name_case_insensitive(self):
        """name.lower() comparison works regardless of case."""
        states = _make_states()

        result = states.find(name="away")
        assert result is not None
        assert result.name() == "Away"

        result2 = states.find(name="AWAY")
        assert result2 is not None
        assert result2.name() == "Away"

    def test_find_no_match_returns_none(self):
        """Non-existent id/name returns None."""
        states = _make_states()

        assert states.find(id=999) is None
        assert states.find(name="nonexistent") is None

    def test_find_no_args_returns_none(self):
        """find() with no arguments returns None."""
        states = _make_states()

        assert states.find() is None
