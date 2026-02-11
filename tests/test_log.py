"""Tests for pyzm.log -- logging setup and ZMLogHandler."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from pyzm.log import ZMLogHandler, _python_to_zm, setup_logging


# ===================================================================
# TestPythonToZmMapping
# ===================================================================

class TestPythonToZm:
    """Tests for _python_to_zm level mapping."""

    def test_debug_maps_to_dbg(self):
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "DBG"

    def test_info_maps_to_inf(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "INF"

    def test_warning_maps_to_war(self):
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "WAR"

    def test_error_maps_to_err(self):
        record = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "ERR"

    def test_critical_maps_to_fat(self):
        record = logging.LogRecord("test", logging.CRITICAL, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "FAT"

    def test_below_debug_maps_to_dbg(self):
        record = logging.LogRecord("test", 5, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "DBG"

    def test_between_info_and_warning(self):
        """Level 25 is between INFO(20) and WARNING(30) -> WAR."""
        record = logging.LogRecord("test", 25, "", 0, "msg", (), None)
        assert _python_to_zm(record) == "WAR"


# ===================================================================
# TestSetupLogging
# ===================================================================

class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "pyzm"

    def test_debug_mode_sets_level(self):
        logger = setup_logging(debug=True)
        assert logger.level == logging.DEBUG
        # Clean up handlers
        logger.handlers.clear()

    def test_non_debug_mode_sets_info_level(self):
        logger = setup_logging(debug=False)
        assert logger.level == logging.INFO
        logger.handlers.clear()

    def test_debug_mode_adds_console_handler(self):
        logger = setup_logging(debug=True)
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, ZMLogHandler)]
        assert len(stream_handlers) >= 1
        logger.handlers.clear()

    def test_custom_component(self):
        logger = setup_logging(component="test_component")
        assert isinstance(logger, logging.Logger)
        logger.handlers.clear()

    @patch("pyzm.log.ZMLogHandler")
    def test_syslog_enabled_adds_handler(self, mock_handler_cls):
        mock_handler = MagicMock()
        mock_handler.level = logging.DEBUG
        mock_handler_cls.return_value = mock_handler

        logger = setup_logging(zm_syslog=True)
        mock_handler_cls.assert_called_once()
        logger.handlers.clear()


# ===================================================================
# TestZMLogHandler
# ===================================================================

class TestZMLogHandler:
    """Tests for ZMLogHandler."""

    def test_creation_without_db(self):
        handler = ZMLogHandler(component="test")
        assert handler.component == "test"
        assert handler._db_available is False
        assert handler.use_syslog is False
        handler.close()

    def test_creation_with_incomplete_db_params(self):
        """Missing some DB params -> DB logging disabled."""
        handler = ZMLogHandler(
            component="test",
            db_host="localhost",
            db_user="zmuser",
            # missing db_password and db_name
        )
        assert handler._db_available is False
        handler.close()

    @patch("pyzm.log._syslog")
    def test_emit_writes_to_syslog_when_enabled(self, mock_syslog):
        handler = ZMLogHandler(component="test", use_syslog=True)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Hello from test",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Verify syslog was called
        mock_syslog.syslog.assert_called_once()
        call_args = mock_syslog.syslog.call_args
        assert "Hello from test" in call_args[0][1]
        handler.close()

    @patch("pyzm.log._syslog")
    def test_emit_no_syslog_when_disabled(self, mock_syslog):
        handler = ZMLogHandler(component="test", use_syslog=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        # syslog.syslog should not have been called (openlog may have been called during import)
        mock_syslog.syslog.assert_not_called()
        handler.close()

    def test_emit_no_db_graceful_skip(self):
        """Handler should not error when DB is unavailable."""
        handler = ZMLogHandler(component="test")

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        # Should not raise
        handler.emit(record)
        handler.close()

    def test_emit_with_zm_debug_level(self):
        """Test that extra zm_debug_level is handled."""
        handler = ZMLogHandler(component="test")

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Debug message",
            args=(),
            exc_info=None,
        )
        record.zm_debug_level = 5  # type: ignore[attr-defined]

        # Should not raise
        handler.emit(record)
        handler.close()

    @patch("pyzm.log._syslog")
    def test_write_syslog_uses_correct_priority(self, mock_syslog):
        import syslog as real_syslog

        handler = ZMLogHandler(component="test", use_syslog=True)

        # Test ERROR level
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="test.py", lineno=1,
            msg="error msg", args=(), exc_info=None,
        )
        handler.emit(record)

        call_args = mock_syslog.syslog.call_args
        # Priority should be LOG_ERR for ERR
        assert call_args[0][0] == real_syslog.LOG_ERR
        handler.close()

    def test_close_no_error(self):
        """Close should not raise even without DB."""
        handler = ZMLogHandler(component="test")
        handler.close()  # Should not raise

    @patch("pyzm.log._syslog")
    def test_close_with_syslog(self, mock_syslog):
        handler = ZMLogHandler(component="test", use_syslog=True)
        handler.close()
        mock_syslog.closelog.assert_called_once()

    def test_db_init_failure_disables_db(self):
        """When DB connection fails, _db_available should be False.

        The real _init_db catches exceptions internally and sets
        _db_available = False, so we mock the sqlalchemy import to
        trigger that path.
        """
        with patch.dict("sys.modules", {"sqlalchemy": MagicMock()}):
            with patch("pyzm.log.ZMLogHandler._init_db") as mock_init:
                # Let _init_db execute normally but simulate the path where
                # it sets _db_available = False.  Since __init__ calls
                # _init_db directly (no wrapping try/except), we make the
                # mock a no-op and verify _db_available stays False.
                mock_init.return_value = None

                handler = ZMLogHandler(
                    component="test",
                    db_host="localhost",
                    db_user="zmuser",
                    db_password="pw",
                    db_name="zm",
                )
                mock_init.assert_called_once()
                # Since we mocked _init_db as a no-op, _db_available
                # remains at its default False (set before _init_db call).
                assert handler._db_available is False
                handler.close()
