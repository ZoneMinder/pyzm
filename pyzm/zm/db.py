"""Direct MySQL connection to the ZoneMinder database.

Reads credentials from ``/etc/zm/zm.conf`` (and ``conf.d/*.conf``) —
the same files that ZM itself uses.
"""

from __future__ import annotations

import configparser
import glob
import logging
import os

logger = logging.getLogger("pyzm.zm")

_CONF_PATH = os.environ.get("PYZM_CONFPATH", "/etc/zm")


def _read_zm_conf(conf_path: str = _CONF_PATH) -> dict[str, str]:
    """Parse ZM config files and return DB credentials."""
    files = sorted(glob.glob(os.path.join(conf_path, "conf.d", "*.conf")))
    files.insert(0, os.path.join(conf_path, "zm.conf"))

    parser = configparser.ConfigParser(
        interpolation=None, inline_comment_prefixes=("#",)
    )
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            parser.read_string("[zm_root]\n" + fh.read())

    section = parser["zm_root"] if parser.has_section("zm_root") else {}
    return {
        "user": section.get("ZM_DB_USER", "zmuser"),
        "password": section.get("ZM_DB_PASS", "zmpass"),
        "host": section.get("ZM_DB_HOST", "localhost"),
        "database": section.get("ZM_DB_NAME", "zm"),
    }


def get_zm_db():
    """Return a ``mysql.connector`` connection to the ZM database, or ``None``."""
    try:
        import mysql.connector
    except ImportError:
        logger.warning("mysql-connector-python not installed, DB access unavailable")
        return None

    creds = _read_zm_conf()
    host = creds["host"]
    port = 3306

    # ZM_DB_HOST can be "hostname:port" or "hostname:/path/to/socket"
    if ":" in host:
        host, suffix = host.split(":", 1)
        if suffix.startswith("/"):
            # Unix socket path — pass as unix_socket
            try:
                return mysql.connector.connect(
                    user=creds["user"],
                    password=creds["password"],
                    database=creds["database"],
                    unix_socket=suffix,
                )
            except mysql.connector.Error as exc:
                logger.warning("DB connect via socket %s failed: %s", suffix, exc)
                return None
        else:
            port = int(suffix)

    try:
        return mysql.connector.connect(
            host=host,
            port=port,
            user=creds["user"],
            password=creds["password"],
            database=creds["database"],
        )
    except mysql.connector.Error as exc:
        logger.warning("DB connect to %s:%s failed: %s", host, port, exc)
        return None
