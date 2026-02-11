"""Authentication manager for ZoneMinder API.

Handles both token-based auth (ZM 1.34+ / API 2.0+) and legacy credential
auth for older ZoneMinder installations.  Tokens are auto-refreshed when
they approach expiry (5-minute grace window).

No global state -- everything is instance-based.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests

logger = logging.getLogger("pyzm.zm")

# Minimum remaining lifetime (seconds) before we proactively refresh.
_REFRESH_GRACE_SECONDS = 5 * 60


def _version_tuple(version: str) -> tuple[int, ...]:
    """Convert a dotted version string to an int tuple for comparison."""
    return tuple(int(p) for p in version.split("."))


class AuthManager:
    """Manages ZM API authentication (token or legacy credentials).

    Parameters
    ----------
    session:
        A ``requests.Session`` used for all HTTP calls.
    api_url:
        Base ZM API URL, e.g. ``https://zm.example.com/zm/api``.
    user:
        ZM username (``None`` when auth is disabled).
    password:
        ZM password (``None`` when auth is disabled).
    token:
        Optional pre-existing refresh/access token to try first.
    """

    def __init__(
        self,
        session: requests.Session,
        api_url: str,
        user: str | None,
        password: str | None,
        token: str | None = None,
    ) -> None:
        self._session = session
        self._api_url = api_url.rstrip("/")
        self._user = user
        self._password = password
        self._initial_token = token

        # Populated after login
        self.api_version: str | None = None
        self.zm_version: str | None = None
        self.auth_enabled: bool = bool(user or token)

        # Token-based auth (API >= 2.0)
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._access_token_expires_at: datetime | None = None
        self._refresh_token_expires_at: datetime | None = None

        # Legacy credential auth (API < 2.0)
        self._legacy_credentials: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Perform a full login against the ZM API.

        On success, populates version info and auth tokens/credentials.
        Raises ``requests.HTTPError`` on failure.
        """
        login_url = f"{self._api_url}/host/login.json"

        if self._initial_token:
            logger.debug("Attempting token login")
            data: dict[str, str] = {"token": self._initial_token}
        elif self._user and self._password:
            logger.debug("Attempting user/password login")
            data = {"user": self._user, "pass": self._password}
        else:
            # No auth -- just grab version info
            logger.debug("Auth disabled; fetching version only")
            self.auth_enabled = False
            self._fetch_version()
            return

        resp = self._session.post(login_url, data=data)

        # If token login returns 401, fall back to user/password
        if resp.status_code == 401 and self._initial_token and self._user and self._password:
            logger.debug("Token login returned 401; falling back to user/password")
            self._initial_token = None
            data = {"user": self._user, "pass": self._password}
            resp = self._session.post(login_url, data=data)

        resp.raise_for_status()
        self._process_login_response(resp.json())

    def apply_auth(
        self, url: str, params: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Inject authentication into *url* / *params* and return them.

        For token auth the token is added as a query parameter.
        For legacy auth the credential string is appended to the URL.

        Returns
        -------
        tuple[str, dict[str, str]]
            The (possibly modified) URL and params dict.
        """
        if params is None:
            params = {}

        if not self.auth_enabled:
            return url, params

        if self._is_token_api():
            params["token"] = self._access_token
        elif self._legacy_credentials:
            sep = "?" if url.lower().endswith(("json", "/")) else "&"
            url = f"{url}{sep}{self._legacy_credentials}"

        return url, params

    def refresh_if_needed(self) -> None:
        """Auto-refresh the access token if it is approaching expiry."""
        if not self.auth_enabled or not self._is_token_api():
            return
        if self._access_token_expires_at is None:
            return

        remaining = (self._access_token_expires_at - datetime.now()).total_seconds()
        if remaining >= _REFRESH_GRACE_SECONDS:
            logger.debug(
                "Access token still valid for %.0f min; no refresh needed",
                remaining / 60,
            )
            return

        logger.debug("Access token expires soon (%.0f s left); refreshing", remaining)
        self._relogin()

    def handle_401(self) -> None:
        """Called when a request gets HTTP 401.  Performs a relogin."""
        logger.debug("Handling 401 -- attempting relogin")
        self._relogin()

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def get_auth_string(self) -> str:
        """Return a bare auth query-string fragment (for portal URLs)."""
        if not self.auth_enabled:
            return ""
        if self._is_token_api():
            return f"token={self._access_token}"
        return self._legacy_credentials or ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_token_api(self) -> bool:
        """True when the ZM API version supports token auth (>= 2.0)."""
        return (
            self.api_version is not None
            and _version_tuple(self.api_version) >= _version_tuple("2.0")
        )

    def _fetch_version(self) -> None:
        """Fetch API/ZM version without authentication."""
        url = f"{self._api_url}/host/getVersion.json"
        resp = self._session.get(url)
        resp.raise_for_status()
        rj = resp.json()
        self.api_version = rj.get("apiversion")
        self.zm_version = rj.get("version")

    def _process_login_response(self, rj: dict) -> None:
        """Extract tokens / credentials from a login JSON response."""
        self.api_version = rj.get("apiversion")
        self.zm_version = rj.get("version")

        if not self.auth_enabled:
            return

        now = datetime.now()

        if self._is_token_api():
            logger.debug("Using token-based auth (API %s)", self.api_version)
            self._access_token = rj.get("access_token", "")

            if rj.get("refresh_token"):
                self._refresh_token = rj["refresh_token"]

            if rj.get("access_token_expires"):
                secs = int(rj["access_token_expires"])
                self._access_token_expires_at = now + timedelta(seconds=secs)
                logger.debug(
                    "Access token expires at %s (%d s)",
                    self._access_token_expires_at, secs,
                )

            if rj.get("refresh_token_expires"):
                secs = int(rj["refresh_token_expires"])
                self._refresh_token_expires_at = now + timedelta(seconds=secs)
                logger.debug(
                    "Refresh token expires at %s (%d s)",
                    self._refresh_token_expires_at, secs,
                )
        else:
            logger.info(
                "Using legacy credential auth (API %s). "
                "Upgrading ZM to 1.34+ is recommended.",
                self.api_version,
            )
            self._legacy_credentials = rj.get("credentials", "")
            if rj.get("append_password") == "1" and self._password:
                self._legacy_credentials += self._password

    def _relogin(self) -> None:
        """Re-authenticate, preferring the refresh token if still valid."""
        if self._is_token_api() and self._refresh_token_expires_at is not None:
            remaining = (self._refresh_token_expires_at - datetime.now()).total_seconds()
            if remaining >= _REFRESH_GRACE_SECONDS:
                logger.debug(
                    "Using refresh token (%.0f min remaining)", remaining / 60,
                )
                self._initial_token = self._refresh_token
            else:
                logger.debug(
                    "Refresh token too close to expiry (%.0f s); "
                    "using user/password",
                    remaining,
                )
                self._initial_token = None

        self.login()
