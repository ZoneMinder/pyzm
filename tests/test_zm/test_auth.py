"""Tests for pyzm.zm.auth -- AuthManager authentication handling."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pyzm.zm.auth import AuthManager, _REFRESH_GRACE_SECONDS, _version_tuple


# ===================================================================
# Helpers
# ===================================================================

def _make_login_response_token(
    api_version: str = "2.0.0",
    zm_version: str = "1.36.12",
    access_token: str = "test_access_token",
    refresh_token: str = "test_refresh_token",
    access_token_expires: int = 3600,
    refresh_token_expires: int = 86400,
) -> dict:
    """Build a realistic token-based login JSON response."""
    return {
        "apiversion": api_version,
        "version": zm_version,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "access_token_expires": str(access_token_expires),
        "refresh_token_expires": str(refresh_token_expires),
    }


def _make_login_response_legacy(
    api_version: str = "1.0.0",
    zm_version: str = "1.32.3",
    credentials: str = "auth=abc123def456",
    append_password: str = "0",
) -> dict:
    """Build a realistic legacy-auth login JSON response."""
    return {
        "apiversion": api_version,
        "version": zm_version,
        "credentials": credentials,
        "append_password": append_password,
    }


def _make_mock_session() -> MagicMock:
    """Create a mock requests.Session."""
    return MagicMock()


# ===================================================================
# TestVersionTuple
# ===================================================================

class TestVersionTuple:
    def test_simple(self):
        assert _version_tuple("2.0") == (2, 0)

    def test_three_parts(self):
        assert _version_tuple("1.36.12") == (1, 36, 12)

    def test_comparison(self):
        assert _version_tuple("2.0") >= _version_tuple("2.0")
        assert _version_tuple("2.0.1") > _version_tuple("2.0")
        assert _version_tuple("1.99") < _version_tuple("2.0")


# ===================================================================
# TestAuthManager - Token Auth
# ===================================================================

class TestAuthManagerTokenLogin:
    """Tests for token-based auth (API >= 2.0)."""

    def test_token_login_populates_tokens(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        assert auth.api_version == "2.0.0"
        assert auth.zm_version == "1.36.12"
        assert auth._access_token == "test_access_token"
        assert auth._refresh_token == "test_refresh_token"
        assert auth._access_token_expires_at is not None
        assert auth._refresh_token_expires_at is not None
        assert auth.auth_enabled is True

    def test_token_login_with_initial_token(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user=None,
            password=None,
            token="existing_token",
        )
        auth.login()

        # Should have posted with token
        call_args = session.post.call_args
        assert call_args[1]["data"]["token"] == "existing_token"

    def test_token_login_fallback_on_401(self):
        """When token login returns 401, falls back to user/password."""
        session = _make_mock_session()

        # First call returns 401, second returns 200
        resp_401 = MagicMock()
        resp_401.status_code = 401

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = _make_login_response_token()

        session.post.side_effect = [resp_401, resp_ok]

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
            token="bad_token",
        )
        auth.login()

        # Should have called post twice
        assert session.post.call_count == 2
        # Second call should use user/password
        second_call = session.post.call_args
        assert second_call[1]["data"] == {"user": "admin", "pass": "secret"}


# ===================================================================
# TestAuthManager - Legacy Auth
# ===================================================================

class TestAuthManagerLegacyLogin:
    """Tests for legacy credential auth (API < 2.0)."""

    def test_legacy_login_populates_credentials(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_legacy()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        assert auth.api_version == "1.0.0"
        assert auth.zm_version == "1.32.3"
        assert auth._legacy_credentials == "auth=abc123def456"
        assert auth._access_token == ""
        assert auth.auth_enabled is True

    def test_legacy_login_append_password(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_legacy(
            credentials="auth=abc123",
            append_password="1",
        )
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="s3cret",
        )
        auth.login()

        assert auth._legacy_credentials == "auth=abc123s3cret"


# ===================================================================
# TestAuthManager - No Auth
# ===================================================================

class TestAuthManagerNoAuth:
    """Tests for auth-disabled flow."""

    def test_no_auth_fetches_version(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "apiversion": "2.0.0",
            "version": "1.36.0",
        }
        session.get.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user=None,
            password=None,
        )
        auth.login()

        assert auth.auth_enabled is False
        assert auth.api_version == "2.0.0"
        assert auth.zm_version == "1.36.0"
        # Should have called GET for version, not POST for login
        session.get.assert_called_once()
        session.post.assert_not_called()


# ===================================================================
# TestAuthManager - apply_auth
# ===================================================================

class TestAuthManagerApplyAuth:
    """Tests for apply_auth method."""

    def test_apply_auth_token_api(self):
        """Token is added to params for API >= 2.0."""
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        url, params = auth.apply_auth(
            "https://zm.example.com/zm/api/monitors.json", {}
        )
        assert params["token"] == "test_access_token"
        # URL should be unchanged
        assert url == "https://zm.example.com/zm/api/monitors.json"

    def test_apply_auth_legacy(self):
        """Credentials appended to URL for legacy auth."""
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_legacy()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        url, params = auth.apply_auth(
            "https://zm.example.com/zm/api/monitors.json", {}
        )
        assert "auth=abc123def456" in url
        assert "token" not in params

    def test_apply_auth_disabled(self):
        """No auth modifications when auth is disabled."""
        session = _make_mock_session()
        resp = MagicMock()
        resp.json.return_value = {"apiversion": "2.0.0", "version": "1.36.0"}
        session.get.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user=None,
            password=None,
        )
        auth.login()

        url, params = auth.apply_auth(
            "https://zm.example.com/zm/api/monitors.json", {}
        )
        assert url == "https://zm.example.com/zm/api/monitors.json"
        assert params == {}

    def test_apply_auth_none_params(self):
        """Params=None should be converted to empty dict."""
        session = _make_mock_session()
        resp = MagicMock()
        resp.json.return_value = {"apiversion": "2.0.0", "version": "1.36.0"}
        session.get.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user=None,
            password=None,
        )
        auth.login()

        url, params = auth.apply_auth(
            "https://zm.example.com/zm/api/test.json", None
        )
        assert isinstance(params, dict)


# ===================================================================
# TestAuthManager - refresh_if_needed
# ===================================================================

class TestAuthManagerRefresh:
    """Tests for refresh_if_needed method."""

    def test_refresh_not_needed_when_token_still_valid(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token(
            access_token_expires=7200,
        )
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        # Reset call count after login
        session.post.reset_mock()

        auth.refresh_if_needed()
        # Should NOT have called post again
        session.post.assert_not_called()

    def test_refresh_triggered_when_token_about_to_expire(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token(
            access_token_expires=3600,
        )
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        # Simulate token about to expire
        auth._access_token_expires_at = datetime.now() + timedelta(seconds=60)

        session.post.reset_mock()
        auth.refresh_if_needed()
        # Should have called _relogin -> login -> post
        assert session.post.call_count >= 1

    def test_refresh_skipped_when_auth_disabled(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.json.return_value = {"apiversion": "2.0.0", "version": "1.36.0"}
        session.get.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user=None,
            password=None,
        )
        auth.login()
        session.post.reset_mock()

        auth.refresh_if_needed()
        session.post.assert_not_called()

    def test_refresh_skipped_for_legacy_auth(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_legacy()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()
        session.post.reset_mock()

        auth.refresh_if_needed()
        session.post.assert_not_called()


# ===================================================================
# TestAuthManager - handle_401
# ===================================================================

class TestAuthManagerHandle401:
    """Tests for handle_401 method."""

    def test_handle_401_triggers_relogin(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        session.post.reset_mock()
        # New response for relogin
        new_resp = MagicMock()
        new_resp.status_code = 200
        new_resp.json.return_value = _make_login_response_token(
            access_token="new_access_token"
        )
        session.post.return_value = new_resp

        auth.handle_401()

        assert session.post.call_count >= 1
        assert auth._access_token == "new_access_token"


# ===================================================================
# TestAuthManager - get_auth_string
# ===================================================================

class TestAuthManagerGetAuthString:
    """Tests for get_auth_string method."""

    def test_token_auth_string(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_token()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        s = auth.get_auth_string()
        assert s == "token=test_access_token"

    def test_legacy_auth_string(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _make_login_response_legacy()
        session.post.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user="admin",
            password="secret",
        )
        auth.login()

        s = auth.get_auth_string()
        assert s == "auth=abc123def456"

    def test_no_auth_returns_empty(self):
        session = _make_mock_session()
        resp = MagicMock()
        resp.json.return_value = {"apiversion": "2.0.0", "version": "1.36.0"}
        session.get.return_value = resp

        auth = AuthManager(
            session=session,
            api_url="https://zm.example.com/zm/api",
            user=None,
            password=None,
        )
        auth.login()

        assert auth.get_auth_string() == ""
