"""Tests for ConsoleLog and Base classes — standalone, no API overlap."""

from unittest.mock import patch
from io import StringIO

import pytest

from pyzm.helpers.Base import Base, ConsoleLog


@pytest.mark.unit
class TestConsoleLog:

    def test_debug_respects_level(self):
        """Debug messages above the set level are suppressed."""
        log = ConsoleLog()
        log.set_level(2)

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            log.Debug(1, "visible")
            log.Debug(2, "also visible")
            log.Debug(3, "suppressed")

        output = mock_out.getvalue()
        assert "visible" in output
        assert "also visible" in output
        assert "suppressed" not in output

    def test_debug_level_getter(self):
        """get_level returns current level."""
        log = ConsoleLog()
        log.set_level(3)
        assert log.get_level() == 3

    def test_fatal_calls_exit(self, no_exit):
        """Fatal prints message then calls exit(-1)."""
        log = ConsoleLog()

        with patch("sys.stdout", new_callable=StringIO):
            log.Fatal("critical error")

        no_exit.assert_called_once_with(-1)

    def test_panic_calls_exit(self, no_exit):
        """Panic prints message then calls exit(-2)."""
        log = ConsoleLog()

        with patch("sys.stdout", new_callable=StringIO):
            log.Panic("panic error")

        no_exit.assert_called_once_with(-2)


@pytest.mark.unit
class TestBase:

    def test_base_instantiation(self):
        """Base class can be instantiated."""
        b = Base()
        assert b is not None
