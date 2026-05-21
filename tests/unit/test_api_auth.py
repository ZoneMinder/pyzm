"""Tests for ZMApi authentication flows."""

import datetime
from unittest.mock import patch

import pytest
import responses

from pyzm.api import ZMApi


@pytest.mark.unit
class TestLoginJWT:
    """JWT token-based login (ZM API >= 2.0)."""

    @responses.activate
    def test_login_jwt_success(self, zm_options, login_success_response):
        """POST login with user/pass stores access and refresh tokens."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        api = ZMApi(options=zm_options.copy())

        assert api.authenticated is True
        assert api.auth_enabled is True
        assert api.access_token == "test_access_token_abc123"
        assert api.refresh_token == "test_refresh_token_xyz789"
        assert api.access_token_expires == 3600
        assert api.refresh_token_expires == 86400
        assert api.access_token_datetime is not None
        assert api.refresh_token_datetime is not None
        assert api.api_version == "2.0"
        assert api.zm_version == "1.36.32"

    @responses.activate
    def test_login_jwt_stores_token_expiry_datetimes(
        self, zm_options, login_success_response
    ):
        """Token expiry datetimes are in the future."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        now = datetime.datetime.now()
        api = ZMApi(options=zm_options.copy())

        assert api.access_token_datetime > now
        assert api.refresh_token_datetime > now


@pytest.mark.unit
class TestLoginLegacy:
    """Legacy credentials-based login (ZM API < 2.0)."""

    @responses.activate
    def test_login_legacy_success(self, login_legacy_response):
        """Legacy login stores credentials string."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_legacy_response,
            status=200,
        )

        options = {
            "apiurl": "https://zm.example.com/zm/api",
            "portalurl": "https://zm.example.com/zm",
            "user": "admin",
            "password": "secret",
            "disable_ssl_cert_check": True,
        }
        api = ZMApi(options=options)

        assert api.authenticated is True
        assert api.legacy_credentials == "auth=abc123hash"
        assert api.api_version == "1.0"

    @responses.activate
    def test_login_legacy_append_password(self):
        """When append_password is '1', password is appended to credentials."""
        legacy_response = {
            "version": "1.32.3",
            "apiversion": "1.0",
            "credentials": "auth=abc123hash",
            "append_password": "1",
        }
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=legacy_response,
            status=200,
        )

        options = {
            "apiurl": "https://zm.example.com/zm/api",
            "portalurl": "https://zm.example.com/zm",
            "user": "admin",
            "password": "mypass",
            "disable_ssl_cert_check": True,
        }
        api = ZMApi(options=options)

        assert api.legacy_credentials == "auth=abc123hashmypass"


@pytest.mark.unit
class TestLoginNoAuth:
    """Login without authentication credentials."""

    @responses.activate
    def test_login_no_auth(self, zm_options_no_auth, version_response):
        """No user/password triggers GET to version endpoint."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/getVersion.json",
            json=version_response,
            status=200,
        )

        api = ZMApi(options=zm_options_no_auth.copy())

        assert api.authenticated is True
        assert api.auth_enabled is False
        assert api.api_version == "2.0"
        assert api.zm_version == "1.36.32"


@pytest.mark.unit
class TestLoginFailure:
    """Login error handling."""

    @responses.activate
    def test_login_failure_raises(self, zm_options):
        """401 from login raises HTTPError."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json={"error": "Unauthorized"},
            status=401,
        )

        import requests

        with pytest.raises(requests.exceptions.HTTPError):
            ZMApi(options=zm_options.copy())

    @responses.activate
    def test_login_token_fallback(self, zm_options_token, login_success_response):
        """Token auth 401 falls back to user/password."""
        # First call with token returns 401
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json={"error": "Token revoked"},
            status=401,
        )
        # Second call with user/pass succeeds
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        api = ZMApi(options=zm_options_token.copy())

        assert api.authenticated is True
        assert api.access_token == "test_access_token_abc123"
        # Two POST requests made
        assert len(responses.calls) == 2


@pytest.mark.unit
class TestPortalURL:
    """Portal URL guessing logic."""

    @responses.activate
    def test_portal_url_guessed(self, login_success_response):
        """When only apiurl is provided, portal URL is derived by stripping /api."""
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        options = {
            "apiurl": "https://zm.example.com/zm/api",
            "user": "admin",
            "password": "secret",
            "disable_ssl_cert_check": True,
        }
        api = ZMApi(options=options)

        assert api.portal_url == "https://zm.example.com/zm"


@pytest.mark.unit
class TestTokenRefresh:
    """Token refresh and relogin logic."""

    def test_token_no_refresh_when_fresh(self, zm_api):
        """Token with >5min remaining skips refresh."""
        # Set expiry far in the future
        zm_api.access_token_datetime = datetime.datetime.now() + datetime.timedelta(
            hours=1
        )
        zm_api.refresh_token_datetime = datetime.datetime.now() + datetime.timedelta(
            hours=24
        )

        # Should not raise or trigger relogin
        zm_api._refresh_tokens_if_needed()
        # Token unchanged
        assert zm_api.access_token == "test_access_token_abc123"

    @responses.activate
    def test_token_refresh_when_near_expiry(
        self, zm_api, login_success_response
    ):
        """_refresh_tokens_if_needed triggers relogin when <5min remaining."""
        # Set access token to expire in 2 minutes
        zm_api.access_token_datetime = datetime.datetime.now() + datetime.timedelta(
            minutes=2
        )
        # Refresh token still valid
        zm_api.refresh_token_datetime = datetime.datetime.now() + datetime.timedelta(
            hours=24
        )

        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        zm_api._refresh_tokens_if_needed()
        assert len(responses.calls) == 1

    def test_token_refresh_skipped_when_no_expiry(self, zm_api):
        """No expiry set means no refresh attempt."""
        zm_api.access_token_expires = None
        zm_api.refresh_token_expires = None

        # Should return without doing anything
        zm_api._refresh_tokens_if_needed()

    @responses.activate
    def test_relogin_uses_refresh_token(self, zm_api, login_success_response):
        """_relogin prefers refresh token when it has >5min remaining."""
        zm_api.refresh_token_datetime = datetime.datetime.now() + datetime.timedelta(
            hours=24
        )

        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        zm_api._relogin()

        # The options should have been updated to use refresh token
        assert zm_api.options["token"] == "test_refresh_token_xyz789"

    @responses.activate
    def test_relogin_uses_credentials_when_refresh_expired(
        self, zm_api, login_success_response
    ):
        """_relogin falls back to user/pass when refresh token near expiry."""
        zm_api.refresh_token_datetime = datetime.datetime.now() + datetime.timedelta(
            seconds=30
        )

        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )

        zm_api._relogin()

        # Token should have been cleared to force user/pass login
        assert zm_api.options.get("token") is None
