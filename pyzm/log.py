"""
pyzm.log -- ZoneMinder-native logging via stdlib logging.

    from pyzm.log import setup_zm_logging
    adapter = setup_zm_logging(name="zmesdetect_m1")
    adapter.Debug(1, "Hello from pyzm")
    adapter.Info("Something happened")
"""
from __future__ import annotations

import configparser as _configparser
import datetime as _datetime
import glob as _glob_mod
import grp as _grp
import logging
import logging.handlers
import os
import pwd as _pwd
import signal as _signal
import syslog as _syslog
import sys as _sys
import time
from typing import Any

__all__ = ["ZMLogAdapter", "setup_zm_logging"]

# ZM level mapping: DBG=1, INF=0, WAR=-1, ERR=-2, FAT=-3
_ZM_LEVELS: dict[str, int] = {"DBG": 1, "INF": 0, "WAR": -1, "ERR": -2, "FAT": -3}

_SYSLOG_PRI: dict[str, int] = {
    "DBG": _syslog.LOG_DEBUG, "INF": _syslog.LOG_INFO,
    "WAR": _syslog.LOG_WARNING, "ERR": _syslog.LOG_ERR, "FAT": _syslog.LOG_ERR,
}


def _python_to_zm(record: logging.LogRecord) -> str:
    """Map a Python log level to a ZM level code."""
    if record.levelno <= logging.DEBUG:
        return "DBG"
    if record.levelno <= logging.INFO:
        return "INF"
    if record.levelno <= logging.WARNING:
        return "WAR"
    if record.levelno <= logging.ERROR:
        return "ERR"
    return "FAT"



# ===================================================================
# ZM-native logging -- replaces pyzm.ZMLog without SQLAlchemy
# ===================================================================

def _read_zm_conf_full(conf_path: str = "/etc/zm") -> dict[str, str]:
    """Parse ZM config files and return DB creds, web user/group, and log path."""
    files = sorted(_glob_mod.glob(os.path.join(conf_path, "conf.d", "*.conf")))
    files.insert(0, os.path.join(conf_path, "zm.conf"))

    parser = _configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#",)
    )
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            parser.read_string("[zm_root]\n" + fh.read())

    section = parser["zm_root"] if parser.has_section("zm_root") else {}
    return {
        "dbuser": section.get("ZM_DB_USER", "zmuser"),
        "dbpassword": section.get("ZM_DB_PASS", "zmpass"),
        "dbhost": section.get("ZM_DB_HOST", "localhost"),
        "dbname": section.get("ZM_DB_NAME", "zm"),
        "webuser": section.get("ZM_WEB_USER", "www-data"),
        "webgroup": section.get("ZM_WEB_GROUP", "www-data"),
        "logpath": section.get("ZM_PATH_LOGS", "/var/log/zm"),
    }


def _read_zm_db_log_config(
    host: str, user: str, password: str, database: str,
) -> dict[str, str]:
    """Read ZM log-level settings from the Config table via mysql.connector."""
    try:
        import mysql.connector
    except ImportError:
        return {}

    connect_kw: dict[str, Any] = {
        "user": user, "password": password, "database": database,
    }
    db_host = host
    if ":" in db_host:
        db_host, suffix = db_host.split(":", 1)
        if suffix.startswith("/"):
            connect_kw["unix_socket"] = suffix
        else:
            connect_kw["host"] = db_host
            connect_kw["port"] = int(suffix)
    else:
        connect_kw["host"] = db_host

    try:
        conn = mysql.connector.connect(**connect_kw)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Name, Value FROM Config WHERE Name IN ("
            "'ZM_LOG_LEVEL_SYSLOG','ZM_LOG_LEVEL_FILE','ZM_LOG_LEVEL_DATABASE',"
            "'ZM_LOG_DEBUG','ZM_LOG_DEBUG_LEVEL','ZM_LOG_DEBUG_FILE',"
            "'ZM_LOG_DEBUG_TARGET','ZM_SERVER_ID')"
        )
        result = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return result
    except Exception:
        return {}


_ZM_OFF = -5  # ZM log level that disables a handler


def _zm_config_to_handler_level(zm_level: int) -> int:
    """Map a ZM config level integer to a Python logging level."""
    if zm_level >= 1:
        return logging.DEBUG
    if zm_level == 0:
        return logging.INFO
    if zm_level == -1:
        return logging.WARNING
    if zm_level == -2:
        return logging.ERROR
    return logging.CRITICAL


class _ZMDBHandler(logging.Handler):
    """Writes to ZM's ``Logs`` table via ``mysql.connector`` (no SQLAlchemy)."""

    def __init__(
        self,
        component: str,
        server_id: int,
        host: str,
        user: str,
        password: str,
        database: str,
        *,
        level: int = logging.DEBUG,
    ) -> None:
        super().__init__(level=level)
        self._component = component
        self._server_id = server_id
        self._host = host
        self._user = user
        self._password = password
        self._database = database
        self._conn: Any = None
        self._reconnecting: bool = False
        self._connect()

    def _connect_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "user": self._user,
            "password": self._password,
            "database": self._database,
        }
        host = self._host
        if ":" in host:
            host, suffix = host.split(":", 1)
            if suffix.startswith("/"):
                kw["unix_socket"] = suffix
                return kw
            kw["port"] = int(suffix)
        kw["host"] = host
        return kw

    def _connect(self) -> None:
        try:
            import mysql.connector
            self._conn = mysql.connector.connect(**self._connect_kwargs())
        except Exception:
            self._conn = None

    def emit(self, record: logging.LogRecord) -> None:
        # Prevent recursive logging during reconnect attempts
        if self._reconnecting:
            return
        # Ping check (matches Perl's $dbh->ping() before every write)
        if self._conn is not None:
            try:
                self._conn.ping(reconnect=False, attempts=1, delay=0)
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        if self._conn is None:
            self._reconnecting = True
            try:
                self._connect()
            finally:
                self._reconnecting = False
            if self._conn is None:
                return

        zm_code = _python_to_zm(record)
        dbg_lvl = getattr(record, "zm_debug_level", 1)
        # Perl uses DB1..DB9 (not DBG1); Level column stores actual sub-level
        if zm_code == "DBG":
            code = f"DB{dbg_lvl}"
            level_val = dbg_lvl
        else:
            code = zm_code
            level_val = _ZM_LEVELS[zm_code]

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "INSERT INTO Logs "
                "(TimeKey,Component,ServerId,Pid,Level,Code,Message,File,Line) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    time.time(), self._component, self._server_id,
                    os.getpid(), level_val, code,
                    record.getMessage(),
                    os.path.basename(record.pathname), record.lineno,
                ),
            )
            self._conn.commit()
            cursor.close()
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        super().close()


class _ZMFileFormatter(logging.Formatter):
    """Produces ZMLog-style file output matching Perl's Logger.pm format.

    Perl format: ``%x %H:%M:%S.%06d id[pid].CODE [caller:line] [message]``
    """

    def __init__(self, process_name: str) -> None:
        super().__init__()
        self._pname = process_name

    def format(self, record: logging.LogRecord) -> str:
        now = _datetime.datetime.now()
        ts = now.strftime("%x %H:%M:%S")
        usec = now.microsecond
        zm_code = _python_to_zm(record)
        dbg_lvl = getattr(record, "zm_debug_level", 1)
        disp = f"DB{dbg_lvl}" if zm_code == "DBG" else zm_code
        caller = f"{os.path.basename(record.pathname)}:{record.lineno}"
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            tb = self.formatException(record.exc_info)
            msg = f"{msg}\n{tb}"
        return (
            f"{ts}.{usec:06d} {self._pname}[{os.getpid()}].{disp} "
            f"[{caller}] [{msg}]"
        )


class _ZMSyslogFormatter(logging.Formatter):
    """Produces ZMLog-style syslog output matching Perl's Logger.pm.

    Perl format: ``CODE [message]``  (ident/pid added by syslog itself)
    """

    def format(self, record: logging.LogRecord) -> str:
        zm_code = _python_to_zm(record)
        dbg_lvl = getattr(record, "zm_debug_level", 1)
        disp = f"DB{dbg_lvl}" if zm_code == "DBG" else zm_code
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            tb = self.formatException(record.exc_info)
            msg = f"{msg}\n{tb}"
        return f"{disp} [{msg}]"


class ZMLogAdapter:
    """Drop-in replacement for the ``pyzm.ZMLog`` module-level API.

    Provides ``Debug``, ``Info``, ``Warning``, ``Error``, ``Fatal``,
    ``close``, and ``get_config`` methods that match the legacy interface.
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: dict[str, Any],
        process_name: str,
    ) -> None:
        self._logger = logger
        self._config = config
        self._process_name = process_name

    def _matches_debug_target(self, target_str: str) -> bool:
        """Check if this process matches ZM_LOG_DEBUG_TARGET (Perl semantics).

        Perl checks exact match against: id, _id, idRoot, _idRoot, or
        empty string (match all).  ``idRoot`` is the part of the process
        name before the first ``_``.
        """
        pname = self._process_name
        id_root = pname.split("_", 1)[0]
        for t in target_str.split("|"):
            t = t.strip()
            if t == "" or t in (pname, f"_{pname}", id_root, f"_{id_root}"):
                return True
        return False

    # -- public API (matches pyzm.ZMLog) --------------------------------

    def Debug(
        self, level: int = 1, msg: str | None = None, caller: Any = None,
    ) -> None:
        target = self._config.get("log_debug_target", "")
        if target and not self._config.get("dump_console"):
            if not self._matches_debug_target(target):
                return

        if self._config.get("log_debug") and level <= self._config.get(
            "log_level_debug", 1
        ):
            if caller is not None:
                rec = logging.LogRecord(
                    self._logger.name, logging.DEBUG,
                    caller.filename, caller.lineno,
                    msg, (), None,
                )
                rec.zm_debug_level = level  # type: ignore[attr-defined]
                self._logger.handle(rec)
            else:
                self._logger.debug(
                    msg, stacklevel=2, extra={"zm_debug_level": level},
                )

    def Info(self, msg: str | None = None, caller: Any = None) -> None:
        if caller is not None:
            rec = logging.LogRecord(
                self._logger.name, logging.INFO,
                caller.filename, caller.lineno,
                msg, (), None,
            )
            self._logger.handle(rec)
        else:
            self._logger.info(msg, stacklevel=2)

    def Warning(self, msg: str | None = None, caller: Any = None) -> None:
        if caller is not None:
            rec = logging.LogRecord(
                self._logger.name, logging.WARNING,
                caller.filename, caller.lineno,
                msg, (), None,
            )
            self._logger.handle(rec)
        else:
            self._logger.warning(msg, stacklevel=2)

    def Error(self, msg: str | None = None, caller: Any = None) -> None:
        if caller is not None:
            rec = logging.LogRecord(
                self._logger.name, logging.ERROR,
                caller.filename, caller.lineno,
                msg, (), None,
            )
            self._logger.handle(rec)
        else:
            self._logger.error(msg, stacklevel=2)

    def Fatal(self, msg: str | None = None, caller: Any = None) -> None:
        if caller is not None:
            rec = logging.LogRecord(
                self._logger.name, logging.CRITICAL,
                caller.filename, caller.lineno,
                msg, (), None,
            )
            self._logger.handle(rec)
        else:
            self._logger.critical(msg, stacklevel=2)
        self.close()
        _sys.exit(-1)

    def close(self) -> None:
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)

    def get_config(self) -> dict[str, Any]:
        return self._config


def setup_zm_logging(
    name: str | None = None,
    override: dict[str, Any] | None = None,
) -> ZMLogAdapter:
    """Initialize ZM-native logging and return a :class:`ZMLogAdapter`.

    This is the v2 replacement for ``pyzm.ZMLog.init()``.  It reads ZM
    config files, the DB ``Config`` table, and environment variables, then
    attaches appropriate handlers to a stdlib logger.

    Args:
        name: Log component name (e.g. ``"zmesdetect_m1"``).
        override: Dict of config keys that override all other sources.
    """
    if override is None:
        override = {}

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    pid = os.getpid()
    if name:
        process_name = name
    else:
        try:
            import psutil
            process_name = psutil.Process(pid).name()
        except ImportError:
            process_name = (
                os.path.basename(_sys.argv[0]) if _sys.argv else "pyzm"
            )

    defaults: dict[str, Any] = {
        "dbuser": None,
        "webuser": "www-data",
        "webgroup": "www-data",
        "logpath": "/var/log/zm",
        "log_level_syslog": 0,
        "log_level_file": 0,
        "log_level_db": 0,
        "log_debug": 0,
        "log_level_debug": 0,
        "log_debug_target": "",
        "log_debug_file": 0,
        "server_id": 0,
        "dump_console": False,
    }

    config: dict[str, Any] = {
        "conf_path": os.environ.get("PYZM_CONFPATH", "/etc/zm"),
        "dbuser": os.environ.get("PYZM_DBUSER"),
        "dbpassword": os.environ.get("PYZM_DBPASSWORD"),
        "dbhost": os.environ.get("PYZM_DBHOST"),
        "webuser": os.environ.get("PYZM_WEBUSER"),
        "webgroup": os.environ.get("PYZM_WEBGROUP"),
        "dbname": os.environ.get("PYZM_DBNAME"),
        "logpath": os.environ.get("PYZM_LOGPATH"),
        "log_level_syslog": os.environ.get("PYZM_SYSLOGLEVEL"),
        "log_level_file": os.environ.get("PYZM_FILELOGLEVEL"),
        "log_level_db": os.environ.get("PYZM_DBLOGLEVEL"),
        "log_debug": os.environ.get("PYZM_LOGDEBUG"),
        "log_level_debug": os.environ.get("PYZM_LOGDEBUGLEVEL"),
        "log_debug_target": os.environ.get("PYZM_LOGDEBUGTARGET"),
        "log_debug_file": os.environ.get("PYZM_LOGDEBUGFILE"),
        "server_id": os.environ.get("PYZM_SERVERID"),
        "dump_console": os.environ.get("PYZM_DUMPCONSOLE"),
    }

    # Apply defaults for None values
    for key, val in defaults.items():
        if config.get(key) is None and val is not None:
            config[key] = val

    # Round 1 overrides (before DB read)
    for key in override:
        if override[key]:
            config[key] = override[key]

    # Read ZM conf files
    conf_data = _read_zm_conf_full(config["conf_path"])
    for src_key in (
        "dbuser", "dbpassword", "dbhost", "dbname",
        "webuser", "webgroup", "logpath",
    ):
        if not config.get(src_key):
            config[src_key] = conf_data.get(src_key)

    # Read log config from DB
    db_vals: dict[str, str] = {}
    if all(config.get(k) for k in ("dbhost", "dbuser", "dbpassword", "dbname")):
        db_vals = _read_zm_db_log_config(
            config["dbhost"], config["dbuser"],
            config["dbpassword"], config["dbname"],
        )

    if db_vals:
        _db_map = {
            "ZM_LOG_LEVEL_SYSLOG": "log_level_syslog",
            "ZM_LOG_LEVEL_FILE": "log_level_file",
            "ZM_LOG_LEVEL_DATABASE": "log_level_db",
            "ZM_LOG_DEBUG": "log_debug",
            "ZM_LOG_DEBUG_LEVEL": "log_level_debug",
            "ZM_LOG_DEBUG_FILE": "log_debug_file",
            "ZM_LOG_DEBUG_TARGET": "log_debug_target",
            "ZM_SERVER_ID": "server_id",
        }
        for db_key, cfg_key in _db_map.items():
            if db_key in db_vals:
                config[cfg_key] = db_vals[db_key]

    # Round 2 overrides (after DB read)
    for key in override:
        if override[key]:
            config[key] = override[key]

    # Fill remaining defaults
    for key in defaults:
        if not config.get(key):
            config[key] = defaults[key]

    # Ensure numeric types
    for k in (
        "log_level_syslog", "log_level_file", "log_level_db",
        "log_debug", "log_level_debug", "server_id",
    ):
        config[k] = int(config[k])

    # --- Apply ZM_LOG_DEBUG_FILE override (Perl behavior) ---
    # When ZM_LOG_DEBUG_FILE is set, it overrides the log file path and
    # raises the file logging level to the debug level.
    if config.get("log_debug") and config.get("log_debug_file") and str(config["log_debug_file"]).strip():
        config["log_level_file"] = max(
            int(config["log_level_file"]),
            int(config.get("log_level_debug", 1)),
        )

    # --- Create stdlib logger and attach handlers ---
    logger = logging.getLogger("pyzm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    if config["log_level_file"] > _ZM_OFF:
        # ZM_LOG_DEBUG_FILE overrides the default log path when set
        debug_file = config.get("log_debug_file")
        if config.get("log_debug") and debug_file and str(debug_file).strip():
            log_fname = str(debug_file).strip()
        else:
            n = os.path.split(process_name)[1].split(".")[0]
            log_fname = os.path.join(config["logpath"], n + ".log")
        try:
            fh = logging.handlers.WatchedFileHandler(log_fname)
            fh.setFormatter(_ZMFileFormatter(process_name))
            fh.setLevel(
                _zm_config_to_handler_level(config["log_level_file"])
            )
            logger.addHandler(fh)
            try:
                uid = _pwd.getpwnam(config["webuser"]).pw_uid
                gid = _grp.getgrnam(config["webgroup"]).gr_gid
                os.chown(log_fname, uid, gid)
            except (KeyError, OSError):
                pass
        except OSError:
            pass

    # DB handler
    if config["log_level_db"] > _ZM_OFF and all(
        config.get(k) for k in ("dbhost", "dbuser", "dbpassword", "dbname")
    ):
        dbh = _ZMDBHandler(
            component=process_name,
            server_id=config["server_id"],
            host=config["dbhost"],
            user=config["dbuser"],
            password=config["dbpassword"],
            database=config["dbname"],
        )
        dbh.setLevel(_zm_config_to_handler_level(config["log_level_db"]))
        logger.addHandler(dbh)

    # Syslog handler -- Perl uses facility=local1 and ident=process_name
    if config["log_level_syslog"] > _ZM_OFF:
        try:
            sh = logging.handlers.SysLogHandler(
                address="/dev/log",
                facility=logging.handlers.SysLogHandler.LOG_LOCAL1,
            )
            sh.setFormatter(_ZMSyslogFormatter())
            sh.setLevel(
                _zm_config_to_handler_level(config["log_level_syslog"])
            )
            # Set the syslog ident so messages show as "pname[pid]: ..."
            sh.ident = f"{process_name}[{os.getpid()}]: "
            logger.addHandler(sh)
        except OSError:
            pass

    # Console handler
    if config.get("dump_console"):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(_ZMFileFormatter(process_name))
        logger.addHandler(ch)

    # SIGHUP -> reopen file handler (log rotation)
    def _sighup(sig: int, frame: Any) -> None:
        for h in logger.handlers[:]:
            if isinstance(h, logging.handlers.WatchedFileHandler):
                h.close()
                logger.removeHandler(h)
        if config["log_level_file"] > _ZM_OFF:
            _n = os.path.split(process_name)[1].split(".")[0]
            _fname = os.path.join(config["logpath"], _n + ".log")
            try:
                new_fh = logging.handlers.WatchedFileHandler(_fname)
                new_fh.setFormatter(_ZMFileFormatter(process_name))
                new_fh.setLevel(
                    _zm_config_to_handler_level(config["log_level_file"])
                )
                logger.addHandler(new_fh)
            except OSError:
                pass

    # SIGUSR1 -> increment log level (more verbose)
    # SIGUSR2 -> decrement log level (less verbose)
    # Matches Perl's logSetSignal() behavior.
    adapter = ZMLogAdapter(logger, config, process_name)

    def _sigusr1(sig: int, frame: Any) -> None:
        for h in logger.handlers:
            cur = h.level
            if cur > logging.DEBUG:
                h.setLevel(cur - 10)

    def _sigusr2(sig: int, frame: Any) -> None:
        for h in logger.handlers:
            cur = h.level
            if cur < logging.CRITICAL:
                h.setLevel(cur + 10)

    try:
        _signal.signal(_signal.SIGHUP, _sighup)
        _signal.signal(_signal.SIGUSR1, _sigusr1)
        _signal.signal(_signal.SIGUSR2, _sigusr2)
    except (OSError, ValueError):
        pass

    return adapter
