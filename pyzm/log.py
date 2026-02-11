"""
pyzm.log -- stdlib logging with an optional ZoneMinder handler.

    from pyzm.log import setup_logging, ZMLogHandler
    logger = setup_logging(debug=True, zm_db_host="localhost",
                           zm_db_user="zmuser", zm_db_password="pw",
                           zm_db_name="zm")
    logger.info("Hello from pyzm")
    logger.debug("Detail", extra={"zm_debug_level": 3})
"""
from __future__ import annotations

import logging
import os
import syslog as _syslog
import time
from typing import Any

__all__ = ["ZMLogHandler", "setup_logging"]

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


class ZMLogHandler(logging.Handler):
    """A ``logging.Handler`` that writes to ZM's Logs DB table and/or syslog.

    All DB and syslog parameters are optional.  If DB credentials are
    incomplete the handler silently skips DB writes.  Pass
    ``extra={"zm_debug_level": N}`` (1-9) on debug messages to set
    the ZM debug sub-level.
    """

    def __init__(
        self,
        component: str = "pyzm",
        *,
        db_host: str | None = None,
        db_user: str | None = None,
        db_password: str | None = None,
        db_name: str | None = None,
        db_driver: str = "mysql+mysqlconnector",
        use_syslog: bool = False,
        level: int = logging.DEBUG,
    ) -> None:
        super().__init__(level=level)
        self.component = component
        self.use_syslog = use_syslog

        self._db_engine: Any = None
        self._db_conn: Any = None
        self._log_table: Any = None
        self._db_available = False

        if all(v is not None for v in (db_host, db_user, db_password, db_name)):
            self._init_db(db_driver, db_user, db_password, db_host, db_name)  # type: ignore[arg-type]

        if self.use_syslog:
            _syslog.openlog(component, _syslog.LOG_PID)

    def _init_db(self, driver: str, user: str, pw: str, host: str, name: str) -> None:
        try:
            from sqlalchemy import MetaData, create_engine
            url = f"{driver}://{user}:{pw}@{host}/{name}"
            self._db_engine = create_engine(url, pool_recycle=3600)
            meta = MetaData()
            meta.reflect(bind=self._db_engine, only=["Logs"])
            self._log_table = meta.tables["Logs"]
            self._db_conn = self._db_engine.connect()
            self._db_available = True
        except Exception as exc:
            self._db_available = False
            logging.getLogger(__name__).warning(
                "ZMLogHandler: DB logging disabled -- %s", exc
            )

    def _write_db(self, record: logging.LogRecord, zm_code: str) -> None:
        if not self._db_available:
            return
        dbg_lvl = getattr(record, "zm_debug_level", 1)
        try:
            cmd = self._log_table.insert().values(
                TimeKey=time.time(),
                Component=self.component,
                ServerId=0,
                Pid=os.getpid(),
                Level=_ZM_LEVELS[zm_code],
                Code=f"{zm_code}{dbg_lvl}" if zm_code == "DBG" else zm_code,
                Message=record.getMessage(),
                File=os.path.basename(record.pathname),
                Line=record.lineno,
            )
            self._db_conn.execute(cmd)
        except Exception as exc:
            self._db_available = False
            logging.getLogger(__name__).warning(
                "ZMLogHandler: DB write failed, disabling -- %s", exc
            )

    def _write_syslog(self, record: logging.LogRecord, zm_code: str) -> None:
        pri = _SYSLOG_PRI.get(zm_code, _syslog.LOG_DEBUG)
        _syslog.syslog(pri, f"{zm_code} [{self.component}] [{record.getMessage()}]")

    def emit(self, record: logging.LogRecord) -> None:
        zm_code = _python_to_zm(record)
        try:
            if self._db_available:
                self._write_db(record, zm_code)
            if self.use_syslog:
                self._write_syslog(record, zm_code)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:
                pass
        if self._db_engine is not None:
            try:
                self._db_engine.dispose()
            except Exception:
                pass
        if self.use_syslog:
            try:
                _syslog.closelog()
            except Exception:
                pass
        super().close()


def setup_logging(
    *,
    debug: bool = False,
    component: str = "pyzm",
    zm_db_host: str | None = None,
    zm_db_user: str | None = None,
    zm_db_password: str | None = None,
    zm_db_name: str | None = None,
    zm_db_driver: str = "mysql+mysqlconnector",
    zm_syslog: bool = False,
) -> logging.Logger:
    """Create and return a configured ``pyzm`` logger.

    - *debug=True* adds a console ``StreamHandler`` and sets level to DEBUG.
    - DB params (all four required) enable writing to ZM's Logs table.
    - *zm_syslog=True* enables syslog output via :class:`ZMLogHandler`.
    """
    logger = logging.getLogger("pyzm")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if debug:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(console)

    want_db = all(
        v is not None
        for v in (zm_db_host, zm_db_user, zm_db_password, zm_db_name)
    )
    if want_db or zm_syslog:
        logger.addHandler(ZMLogHandler(
            component=component,
            db_host=zm_db_host,
            db_user=zm_db_user,
            db_password=zm_db_password,
            db_name=zm_db_name,
            db_driver=zm_db_driver,
            use_syslog=zm_syslog,
        ))

    return logger
