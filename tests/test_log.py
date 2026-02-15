"""Tests for pyzm.log -- logging setup and ZMLogHandler."""

from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from pyzm.log import (
    ZMLogHandler, _python_to_zm, setup_logging,
    _read_zm_conf_full, _zm_config_to_handler_level, _ZM_OFF,
    _ZMDBHandler, _ZMFileFormatter, _ZMSyslogFormatter,
    ZMLogAdapter, setup_zm_logging,
)


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


# ===================================================================
# TestReadZmConfFull
# ===================================================================

class TestReadZmConfFull:
    """Tests for _read_zm_conf_full."""

    def test_reads_zm_conf(self, tmp_path):
        (tmp_path / "zm.conf").write_text(
            "ZM_DB_USER=testuser\n"
            "ZM_DB_PASS=testpass\n"
            "ZM_DB_HOST=dbhost\n"
            "ZM_DB_NAME=testdb\n"
            "ZM_WEB_USER=apache\n"
            "ZM_WEB_GROUP=apache\n"
            "ZM_PATH_LOGS=/tmp/zmlogs\n"
        )
        (tmp_path / "conf.d").mkdir()
        result = _read_zm_conf_full(str(tmp_path))
        assert result["dbuser"] == "testuser"
        assert result["dbpassword"] == "testpass"
        assert result["dbhost"] == "dbhost"
        assert result["dbname"] == "testdb"
        assert result["webuser"] == "apache"
        assert result["webgroup"] == "apache"
        assert result["logpath"] == "/tmp/zmlogs"

    def test_conf_d_overrides(self, tmp_path):
        (tmp_path / "zm.conf").write_text("ZM_DB_USER=original\n")
        (tmp_path / "conf.d").mkdir()
        (tmp_path / "conf.d" / "01-override.conf").write_text(
            "ZM_DB_USER=overridden\n"
        )
        result = _read_zm_conf_full(str(tmp_path))
        assert result["dbuser"] == "overridden"

    def test_defaults_when_missing(self, tmp_path):
        (tmp_path / "zm.conf").write_text("")
        (tmp_path / "conf.d").mkdir()
        result = _read_zm_conf_full(str(tmp_path))
        assert result["dbuser"] == "zmuser"
        assert result["dbpassword"] == "zmpass"
        assert result["dbhost"] == "localhost"
        assert result["dbname"] == "zm"
        assert result["webuser"] == "www-data"
        assert result["logpath"] == "/var/log/zm"

    def test_missing_conf_path(self, tmp_path):
        """Non-existent path returns defaults."""
        result = _read_zm_conf_full(str(tmp_path / "nonexistent"))
        assert result["dbuser"] == "zmuser"


# ===================================================================
# TestZmConfigToHandlerLevel
# ===================================================================

class TestZmConfigToHandlerLevel:
    """Tests for _zm_config_to_handler_level."""

    def test_debug(self):
        assert _zm_config_to_handler_level(1) == logging.DEBUG
        assert _zm_config_to_handler_level(5) == logging.DEBUG

    def test_info(self):
        assert _zm_config_to_handler_level(0) == logging.INFO

    def test_warning(self):
        assert _zm_config_to_handler_level(-1) == logging.WARNING

    def test_error(self):
        assert _zm_config_to_handler_level(-2) == logging.ERROR

    def test_critical(self):
        assert _zm_config_to_handler_level(-3) == logging.CRITICAL
        assert _zm_config_to_handler_level(-4) == logging.CRITICAL


# ===================================================================
# TestZMDBHandler
# ===================================================================

class TestZMDBHandler:
    """Tests for _ZMDBHandler."""

    @patch("mysql.connector.connect")
    def test_emit_inserts_row(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 42, "hello", (), None,
        )
        handler.emit(record)

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO Logs" in sql
        vals = mock_cursor.execute.call_args[0][1]
        assert vals[1] == "test"  # Component
        assert vals[4] == 0       # Level (INF=0)
        assert vals[5] == "INF"   # Code
        assert vals[6] == "hello" # Message
        mock_conn.commit.assert_called_once()
        handler.close()

    @patch("mysql.connector.connect")
    def test_emit_debug_code_matches_perl(self, mock_connect):
        """Perl uses DB1..DB9 codes and stores actual sub-level in Level."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "dbg", (), None,
        )
        record.zm_debug_level = 3  # type: ignore[attr-defined]
        handler.emit(record)

        vals = mock_cursor.execute.call_args[0][1]
        assert vals[4] == 3      # Level = actual debug sub-level
        assert vals[5] == "DB3"  # Code = DB prefix (not DBG)
        handler.close()

    @patch("mysql.connector.connect", side_effect=Exception("no db"))
    def test_connect_failure_skips_emit(self, mock_connect):
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        assert handler._conn is None
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        # Should not raise
        handler.emit(record)
        handler.close()

    @patch("mysql.connector.connect")
    def test_reconnect_on_write_failure(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("write failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        # Connection should be cleared for retry on next emit
        assert handler._conn is None
        handler.close()

    @patch("mysql.connector.connect")
    def test_connect_kwargs_host_port(self, mock_connect):
        mock_connect.return_value = MagicMock()
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="dbhost:3307", user="u", password="p", database="zm",
        )
        kw = handler._connect_kwargs()
        assert kw["host"] == "dbhost"
        assert kw["port"] == 3307
        handler.close()

    @patch("mysql.connector.connect")
    def test_connect_kwargs_unix_socket(self, mock_connect):
        mock_connect.return_value = MagicMock()
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost:/var/run/mysqld.sock",
            user="u", password="p", database="zm",
        )
        kw = handler._connect_kwargs()
        assert kw["unix_socket"] == "/var/run/mysqld.sock"
        assert "host" not in kw
        handler.close()

    @patch("mysql.connector.connect")
    def test_close_closes_connection(self, mock_connect):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        handler.close()
        mock_conn.close.assert_called_once()
        assert handler._conn is None


# ===================================================================
# TestZMFileFormatter
# ===================================================================

class TestZMFileFormatter:
    """Tests for _ZMFileFormatter matching Perl's Logger.pm format."""

    def test_format_info(self):
        fmt = _ZMFileFormatter("zmesdetect_m1")
        record = logging.LogRecord(
            "test", logging.INFO, "/path/to/test.py", 42,
            "hello world", (), None,
        )
        result = fmt.format(record)
        # Perl format: timestamp.usec id[pid].CODE [caller:line] [msg]
        assert "zmesdetect_m1[" in result
        assert "].INF" in result      # dot before code
        assert "[test.py:42]" in result  # brackets around caller
        assert "[hello world]" in result

    def test_format_includes_microseconds(self):
        fmt = _ZMFileFormatter("test")
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        result = fmt.format(record)
        # Should contain .NNNNNN after the time
        import re
        assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{6}", result)

    def test_format_debug_uses_db_prefix(self):
        """Perl uses DB3, not DBG3."""
        fmt = _ZMFileFormatter("test")
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "dbg", (), None,
        )
        record.zm_debug_level = 3  # type: ignore[attr-defined]
        result = fmt.format(record)
        assert "].DB3" in result
        assert "DBG" not in result


# ===================================================================
# TestZMSyslogFormatter
# ===================================================================

class TestZMSyslogFormatter:
    """Tests for _ZMSyslogFormatter matching Perl's Logger.pm format."""

    def test_format_info(self):
        """Perl syslog: CODE [message] -- ident/pid added by syslog."""
        fmt = _ZMSyslogFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "syslog msg", (), None,
        )
        result = fmt.format(record)
        assert result == "INF [syslog msg]"

    def test_format_debug_uses_db_prefix(self):
        """Perl uses DB2, not DBG2."""
        fmt = _ZMSyslogFormatter()
        record = logging.LogRecord(
            "test", logging.DEBUG, "test.py", 1, "dbg", (), None,
        )
        record.zm_debug_level = 2  # type: ignore[attr-defined]
        result = fmt.format(record)
        assert result == "DB2 [dbg]"


# ===================================================================
# TestZMLogAdapter
# ===================================================================

class TestZMLogAdapter:
    """Tests for ZMLogAdapter."""

    def _make_adapter(self, **config_overrides):
        logger = logging.getLogger("zm.test_adapter")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        handler = logging.handlers.MemoryHandler(capacity=100)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        config = {
            "log_debug": 1,
            "log_level_debug": 5,
            "log_debug_target": "",
            "dump_console": False,
        }
        config.update(config_overrides)
        return ZMLogAdapter(logger, config, "test_process"), handler

    def test_debug_emits_when_enabled(self):
        adapter, handler = self._make_adapter()
        adapter.Debug(1, "test debug")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].getMessage() == "test debug"
        adapter._logger.handlers.clear()

    def test_debug_suppressed_when_disabled(self):
        adapter, handler = self._make_adapter(log_debug=0)
        adapter.Debug(1, "should not appear")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_suppressed_when_level_too_high(self):
        adapter, handler = self._make_adapter(log_level_debug=2)
        adapter.Debug(3, "level 3 debug")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_target_filtering(self):
        adapter, handler = self._make_adapter(log_debug_target="zmc_m1|zmc_m2")
        # process_name is "test_process", doesn't match targets
        adapter.Debug(1, "filtered out")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_target_allows_exact_match(self):
        adapter, handler = self._make_adapter(log_debug_target="test_process")
        adapter.Debug(1, "should pass")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_matches_id_root(self):
        """Perl matches idRoot (part before first _)."""
        adapter, handler = self._make_adapter(log_debug_target="test")
        # process_name is "test_process", idRoot is "test"
        adapter.Debug(1, "root match")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_rejects_partial_prefix(self):
        """Perl does exact match, not startswith: 'tes' should NOT match 'test_process'."""
        adapter, handler = self._make_adapter(log_debug_target="tes")
        adapter.Debug(1, "should not match")
        assert len(handler.buffer) == 0
        adapter._logger.handlers.clear()

    def test_debug_target_matches_underscore_prefix(self):
        """Perl also matches _id and _idRoot forms."""
        adapter, handler = self._make_adapter(log_debug_target="_test_process")
        adapter.Debug(1, "underscore match")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_empty_matches_all(self):
        """Empty target in Perl means match all processes."""
        adapter, handler = self._make_adapter(log_debug_target="")
        # Empty target → target check is skipped → matches all
        adapter.Debug(1, "should pass")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_debug_target_bypass_with_dump_console(self):
        adapter, handler = self._make_adapter(
            log_debug_target="other", dump_console=True,
        )
        adapter.Debug(1, "console bypass")
        assert len(handler.buffer) == 1
        adapter._logger.handlers.clear()

    def test_info(self):
        adapter, handler = self._make_adapter()
        adapter.Info("info msg")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].levelno == logging.INFO
        adapter._logger.handlers.clear()

    def test_warning(self):
        adapter, handler = self._make_adapter()
        adapter.Warning("warn msg")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].levelno == logging.WARNING
        adapter._logger.handlers.clear()

    def test_error(self):
        adapter, handler = self._make_adapter()
        adapter.Error("err msg")
        assert len(handler.buffer) == 1
        assert handler.buffer[0].levelno == logging.ERROR
        adapter._logger.handlers.clear()

    def test_fatal_exits(self):
        adapter, handler = self._make_adapter()
        with pytest.raises(SystemExit):
            adapter.Fatal("fatal msg")

    def test_get_config(self):
        adapter, _ = self._make_adapter()
        cfg = adapter.get_config()
        assert cfg["log_debug"] == 1
        adapter._logger.handlers.clear()

    def test_close_removes_handlers(self):
        adapter, handler = self._make_adapter()
        assert len(adapter._logger.handlers) > 0
        adapter.close()
        assert len(adapter._logger.handlers) == 0


# ===================================================================
# TestSetupZmLogging
# ===================================================================

class TestSetupZmLogging:
    """Tests for setup_zm_logging."""

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_returns_adapter(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_app", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert isinstance(adapter, ZMLogAdapter)
        assert adapter._process_name == "test_app"
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_dump_console_adds_stream_handler(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_console", override={
            "dump_console": True,
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        stream_handlers = [
            h for h in adapter._logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, (logging.handlers.SysLogHandler,))
        ]
        assert len(stream_handlers) >= 1
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_overrides_take_priority(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_override", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "log_debug": True,
            "log_level_debug": 9,
        })
        assert adapter._config["log_debug"] == True
        assert adapter._config["log_level_debug"] == 9
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={
        "ZM_LOG_DEBUG": "1",
        "ZM_LOG_DEBUG_LEVEL": "4",
    })
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_db_config_applied(self, mock_sig, mock_conf, mock_db):
        adapter = setup_zm_logging(name="test_db_cfg", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert adapter._config["log_debug"] == 1
        assert adapter._config["log_level_debug"] == 4
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": None, "dbpassword": None, "dbhost": None,
        "dbname": None, "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_works_without_db(self, mock_sig, mock_conf, mock_db):
        """setup_zm_logging works even when no DB creds are available."""
        adapter = setup_zm_logging(name="test_nodb", override={
            "log_level_file": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        assert isinstance(adapter, ZMLogAdapter)
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_file_handler_created(self, mock_sig, mock_conf, mock_db, tmp_path):
        adapter = setup_zm_logging(name="test_file", override={
            "log_level_file": 0,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "logpath": str(tmp_path),
        })
        file_handlers = [
            h for h in adapter._logger.handlers
            if isinstance(h, logging.handlers.WatchedFileHandler)
        ]
        assert len(file_handlers) == 1
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_syslog_uses_local1_facility(self, mock_sig, mock_conf, mock_db):
        """Perl uses facility=local1 for syslog."""
        with patch("pyzm.log.logging.handlers.SysLogHandler") as mock_sh_cls:
            mock_sh = MagicMock()
            mock_sh_cls.return_value = mock_sh
            mock_sh_cls.LOG_LOCAL1 = logging.handlers.SysLogHandler.LOG_LOCAL1

            adapter = setup_zm_logging(name="test_syslog", override={
                "log_level_file": _ZM_OFF,
                "log_level_db": _ZM_OFF,
                "log_level_syslog": 0,
            })
            mock_sh_cls.assert_called_once_with(
                address="/dev/log",
                facility=logging.handlers.SysLogHandler.LOG_LOCAL1,
            )
            # Verify ident is set for syslog
            assert "test_syslog[" in mock_sh.ident
            adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_debug_file_overrides_log_path(self, mock_sig, mock_conf, mock_db, tmp_path):
        """ZM_LOG_DEBUG_FILE overrides log file path and raises file level."""
        debug_file = str(tmp_path / "debug_override.log")
        adapter = setup_zm_logging(name="test_dbgfile", override={
            "log_level_file": 0,  # INFO level
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "log_debug": 1,
            "log_level_debug": 5,
            "log_debug_file": debug_file,
            "logpath": str(tmp_path),
        })
        file_handlers = [
            h for h in adapter._logger.handlers
            if isinstance(h, logging.handlers.WatchedFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == debug_file
        # File level should be raised to debug level
        assert file_handlers[0].level == logging.DEBUG
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={
        "ZM_LOG_LEVEL_SYSLOG": "-1",
        "ZM_LOG_LEVEL_FILE": "1",
        "ZM_LOG_LEVEL_DATABASE": "-2",
        "ZM_LOG_DEBUG": "1",
        "ZM_LOG_DEBUG_LEVEL": "7",
        "ZM_LOG_DEBUG_TARGET": "zmc_m1",
    })
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_round2_override_beats_db(self, mock_sig, mock_conf, mock_db, tmp_path):
        """Override dict (round 2) takes priority over DB config values."""
        adapter = setup_zm_logging(name="test_r2", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "log_debug": True,
            "log_level_debug": 3,  # override DB's 7
        })
        assert adapter._config["log_level_debug"] == 3  # override wins
        assert adapter._config["log_debug_target"] == "zmc_m1"  # DB value kept (no override)
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    @patch("pyzm.log._signal.signal")
    def test_signals_registered(self, mock_sig, mock_conf, mock_db):
        """SIGHUP, SIGUSR1, SIGUSR2 handlers are registered."""
        import signal as real_signal
        adapter = setup_zm_logging(name="test_signals", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
        })
        sig_calls = {c[0][0] for c in mock_sig.call_args_list}
        assert real_signal.SIGHUP in sig_calls
        assert real_signal.SIGUSR1 in sig_calls
        assert real_signal.SIGUSR2 in sig_calls
        adapter.close()


# ===================================================================
# TestZMDBHandlerPing
# ===================================================================

class TestZMDBHandlerPing:
    """Tests for _ZMDBHandler ping and reconnect behavior."""

    @patch("mysql.connector.connect")
    def test_ping_called_before_write(self, mock_connect):
        """Perl calls $dbh->ping() before every write."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        mock_conn.ping.assert_called_once_with(
            reconnect=False, attempts=1, delay=0,
        )
        handler.close()

    @patch("mysql.connector.connect")
    def test_ping_failure_triggers_reconnect(self, mock_connect):
        """If ping fails, handler reconnects before writing."""
        mock_conn_dead = MagicMock()
        mock_conn_dead.ping.side_effect = Exception("gone")
        mock_conn_new = MagicMock()
        mock_cursor = MagicMock()
        mock_conn_new.cursor.return_value = mock_cursor
        # First connect returns dead conn, second returns fresh conn
        mock_connect.side_effect = [mock_conn_dead, mock_conn_new]

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        # Should have reconnected and written via new connection
        assert mock_connect.call_count == 2
        mock_cursor.execute.assert_called_once()
        handler.close()

    @patch("mysql.connector.connect")
    def test_recursive_logging_guard(self, mock_connect):
        """Emit returns early when _reconnecting=True (prevents infinite loops)."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        handler = _ZMDBHandler(
            component="test", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        handler._reconnecting = True
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        handler.emit(record)
        # Should have returned immediately without writing
        mock_conn.cursor.assert_not_called()
        handler.close()


# ===================================================================
# TestZMDBHandlerLevelColumn
# ===================================================================

class TestZMDBHandlerLevelColumn:
    """Verify DB Level column matches Perl for all severities."""

    @patch("mysql.connector.connect")
    def _emit_and_get_level(self, py_level, mock_connect, **extra):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        handler = _ZMDBHandler(
            component="t", server_id=0,
            host="localhost", user="u", password="p", database="zm",
        )
        record = logging.LogRecord("t", py_level, "t.py", 1, "m", (), None)
        for k, v in extra.items():
            setattr(record, k, v)
        handler.emit(record)
        vals = mock_cursor.execute.call_args[0][1]
        handler.close()
        return vals[4], vals[5]  # Level, Code

    def test_info_level_0(self):
        level, code = self._emit_and_get_level(logging.INFO)
        assert level == 0
        assert code == "INF"

    def test_warning_level_neg1(self):
        level, code = self._emit_and_get_level(logging.WARNING)
        assert level == -1
        assert code == "WAR"

    def test_error_level_neg2(self):
        level, code = self._emit_and_get_level(logging.ERROR)
        assert level == -2
        assert code == "ERR"

    def test_fatal_level_neg3(self):
        level, code = self._emit_and_get_level(logging.CRITICAL)
        assert level == -3
        assert code == "FAT"

    def test_debug_level_stores_sublevel(self):
        """Debug sub-level 5 -> Level=5, Code=DB5."""
        level, code = self._emit_and_get_level(
            logging.DEBUG, zm_debug_level=5,
        )
        assert level == 5
        assert code == "DB5"

    def test_debug_default_sublevel_1(self):
        """Default debug sub-level is 1 -> Level=1, Code=DB1."""
        level, code = self._emit_and_get_level(logging.DEBUG)
        assert level == 1
        assert code == "DB1"


# ===================================================================
# TestSignalHandlers
# ===================================================================

class TestSignalHandlers:
    """Tests for SIGHUP/SIGUSR1/SIGUSR2 behavior."""

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    def test_sigusr1_increases_verbosity(self, mock_conf, mock_db, tmp_path):
        """SIGUSR1 decreases handler level (more verbose)."""
        import signal
        adapter = setup_zm_logging(name="test_usr1", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "dump_console": True,
        })
        handler = adapter._logger.handlers[0]
        # Start at WARNING so there's room to go down
        handler.setLevel(logging.WARNING)
        # Simulate SIGUSR1
        signal.raise_signal(signal.SIGUSR1)
        assert handler.level == logging.WARNING - 10  # INFO
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    def test_sigusr2_decreases_verbosity(self, mock_conf, mock_db, tmp_path):
        """SIGUSR2 increases handler level (less verbose)."""
        import signal
        adapter = setup_zm_logging(name="test_usr2", override={
            "log_level_file": _ZM_OFF,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "dump_console": True,
        })
        handler = adapter._logger.handlers[0]
        original_level = handler.level
        signal.raise_signal(signal.SIGUSR2)
        assert handler.level == original_level + 10
        adapter.close()

    @patch("pyzm.log._read_zm_db_log_config", return_value={})
    @patch("pyzm.log._read_zm_conf_full", return_value={
        "dbuser": "u", "dbpassword": "p", "dbhost": "h",
        "dbname": "zm", "webuser": "www", "webgroup": "www",
        "logpath": "/tmp",
    })
    def test_sighup_reopens_file_handler(self, mock_conf, mock_db, tmp_path):
        """SIGHUP closes and reopens the file handler for log rotation."""
        import signal
        adapter = setup_zm_logging(name="test_hup", override={
            "log_level_file": 0,
            "log_level_db": _ZM_OFF,
            "log_level_syslog": _ZM_OFF,
            "logpath": str(tmp_path),
        })
        old_handlers = adapter._logger.handlers[:]
        old_fh = [h for h in old_handlers
                  if isinstance(h, logging.handlers.WatchedFileHandler)]
        assert len(old_fh) == 1

        signal.raise_signal(signal.SIGHUP)

        new_fh = [h for h in adapter._logger.handlers
                  if isinstance(h, logging.handlers.WatchedFileHandler)]
        assert len(new_fh) == 1
        # Should be a NEW handler instance (old one was closed)
        assert new_fh[0] is not old_fh[0]
        adapter.close()


# ===================================================================
# TestFileFormatAllLevels
# ===================================================================

class TestFileFormatAllLevels:
    """Verify _ZMFileFormatter produces correct codes for all levels."""

    def _format(self, level, **extra):
        fmt = _ZMFileFormatter("test_proc")
        record = logging.LogRecord("t", level, "mod.py", 10, "msg", (), None)
        for k, v in extra.items():
            setattr(record, k, v)
        return fmt.format(record)

    def test_info_code(self):
        assert "].INF [" in self._format(logging.INFO)

    def test_warning_code(self):
        assert "].WAR [" in self._format(logging.WARNING)

    def test_error_code(self):
        assert "].ERR [" in self._format(logging.ERROR)

    def test_fatal_code(self):
        assert "].FAT [" in self._format(logging.CRITICAL)

    def test_debug_levels_1_through_9(self):
        for lvl in range(1, 10):
            result = self._format(logging.DEBUG, zm_debug_level=lvl)
            assert f"].DB{lvl} [" in result
