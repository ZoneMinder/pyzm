"""E2E tests for ZMApi authentication and basic server info.

All tests are readonly — they only query the server.
"""

import re

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_readonly]


class TestLogin:
    """Verify login succeeds and sets expected attributes."""

    def test_login_succeeds(self, zm_api_live):
        assert zm_api_live.authenticated is True

    def test_login_sets_api_version(self, zm_api_live):
        v = zm_api_live.api_version
        assert v is not None
        assert isinstance(v, str)
        # API version should be a dotted number like "2.0"
        assert re.match(r"^\d+\.\d+", v), f"Unexpected api_version format: {v}"

    def test_login_sets_zm_version(self, zm_api_live):
        v = zm_api_live.zm_version
        assert v is not None
        assert isinstance(v, str)
        # ZM version should look like "1.36.33" or similar
        assert re.match(r"^\d+\.\d+", v), f"Unexpected zm_version format: {v}"


class TestVersion:
    """Verify version() returns proper structure."""

    def test_version_returns_ok_status(self, zm_api_live):
        result = zm_api_live.version()
        assert result["status"] == "ok"

    def test_version_has_expected_keys(self, zm_api_live):
        result = zm_api_live.version()
        assert "api_version" in result
        assert "zm_version" in result
        assert isinstance(result["api_version"], str)
        assert isinstance(result["zm_version"], str)


class TestTimezone:
    """Verify tz() returns Area/Location format."""

    def test_tz_returns_valid_timezone(self, zm_api_live):
        tz = zm_api_live.tz()
        assert tz is not None
        assert isinstance(tz, str)
        # Timezone should be in Area/Location format (e.g. "America/New_York")
        assert "/" in tz, f"Timezone not in Area/Location format: {tz}"


class TestGetAuth:
    """Verify get_auth() returns a usable auth string."""

    def test_get_auth_returns_string(self, zm_api_live):
        auth = zm_api_live.get_auth()
        assert isinstance(auth, str)
        # Should be either "token=..." or "auth=..." depending on API version
        assert auth.startswith("token=") or auth.startswith("auth="), \
            f"Unexpected auth format: {auth}"

    def test_get_auth_nonempty(self, zm_api_live):
        auth = zm_api_live.get_auth()
        assert len(auth) > 6, "Auth string too short to contain a real token"


class TestFreshLogin:
    """Verify a fresh login works (function-scoped, not session-cached)."""

    def test_fresh_login_authenticates(self, zm_api_fresh):
        assert zm_api_fresh.authenticated is True
        assert zm_api_fresh.api_version is not None


class TestBadCredentials:
    """Verify bad credentials raise an error."""

    def test_bad_password_raises(self, zm_options_live):
        from pyzm.api import ZMApi

        bad_opts = zm_options_live.copy()
        bad_opts["password"] = "definitely_wrong_password_xyz"
        with pytest.raises(Exception):
            ZMApi(options=bad_opts)
