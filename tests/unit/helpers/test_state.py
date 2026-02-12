"""Edge-case tests for State helper."""

from unittest.mock import MagicMock

import pytest

from pyzm.helpers.State import State


@pytest.mark.unit
class TestStateEdgeCases:

    def test_active_false(self):
        """IsActive != '1' returns False."""
        state = State(
            state={"State": {"Id": "2", "Name": "away", "IsActive": "0", "Definition": "All active"}},
            api=MagicMock(),
        )
        assert state.active() is False

    def test_definition_none(self):
        """Empty definition returns None."""
        state = State(
            state={"State": {"Id": "3", "Name": "home", "IsActive": "0", "Definition": ""}},
            api=MagicMock(),
        )
        assert state.definition() is None

    def test_definition_with_value(self):
        """Non-empty definition returns the string."""
        state = State(
            state={"State": {"Id": "1", "Name": "default", "IsActive": "1", "Definition": "ZM default"}},
            api=MagicMock(),
        )
        assert state.definition() == "ZM default"
