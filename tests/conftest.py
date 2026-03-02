"""Shared test fixtures for pyzm test suite."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import responses

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "responses")


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Option dicts
# ---------------------------------------------------------------------------

@pytest.fixture
def zm_options():
    """Standard ZM API options dict for JWT auth."""
    return {
        "apiurl": "https://zm.example.com/zm/api",
        "portalurl": "https://zm.example.com/zm",
        "user": "admin",
        "password": "secret",
        "disable_ssl_cert_check": True,
    }


@pytest.fixture
def zm_options_no_auth():
    """ZM API options without authentication."""
    return {
        "apiurl": "https://zm.example.com/zm/api",
        "portalurl": "https://zm.example.com/zm",
        "disable_ssl_cert_check": True,
    }


@pytest.fixture
def zm_options_token():
    """ZM API options with token auth."""
    return {
        "apiurl": "https://zm.example.com/zm/api",
        "portalurl": "https://zm.example.com/zm",
        "user": "admin",
        "password": "secret",
        "token": "existing_token_123",
        "disable_ssl_cert_check": True,
    }


# ---------------------------------------------------------------------------
# JSON response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def login_success_response():
    return _load_fixture("login_success.json")


@pytest.fixture
def login_legacy_response():
    return _load_fixture("login_legacy.json")


@pytest.fixture
def monitors_response():
    return _load_fixture("monitors.json")


@pytest.fixture
def events_response():
    return _load_fixture("events.json")


@pytest.fixture
def states_response():
    return _load_fixture("states.json")


@pytest.fixture
def configs_response():
    return _load_fixture("configs.json")


@pytest.fixture
def version_response():
    return _load_fixture("version.json")


@pytest.fixture
def daemon_status_response():
    return _load_fixture("daemon_status.json")


# ---------------------------------------------------------------------------
# Logger suppression
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def suppress_logger():
    """Replace g.logger with a silent mock to prevent console spam."""
    mock_logger = MagicMock()
    with patch("pyzm.helpers.globals.logger", mock_logger):
        # Also patch the module-level reference that may have been cached
        import pyzm.helpers.globals as g
        original = g.logger
        g.logger = mock_logger
        yield mock_logger
        g.logger = original


# ---------------------------------------------------------------------------
# Prevent exit() calls in ConsoleLog.Fatal / Panic
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_exit():
    """Prevent ConsoleLog.Fatal/Panic from killing the test runner."""
    with patch("builtins.exit") as mock_exit:
        yield mock_exit


# ---------------------------------------------------------------------------
# Pre-authenticated ZMApi factory
# ---------------------------------------------------------------------------

@pytest.fixture
def zm_api(zm_options, login_success_response):
    """Return a pre-authenticated ZMApi instance with login mocked."""
    from pyzm.api import ZMApi

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )
        api = ZMApi(options=zm_options.copy())

    assert api.authenticated is True
    assert api.api_version == "2.0"
    assert api.access_token == "test_access_token_abc123"
    return api


@pytest.fixture
def zm_api_legacy(login_legacy_response):
    """Return a pre-authenticated ZMApi instance using legacy credentials."""
    from pyzm.api import ZMApi

    options = {
        "apiurl": "https://zm.example.com/zm/api",
        "portalurl": "https://zm.example.com/zm",
        "user": "admin",
        "password": "secret",
        "disable_ssl_cert_check": True,
    }

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_legacy_response,
            status=200,
        )
        api = ZMApi(options=options.copy())

    assert api.authenticated is True
    assert api.api_version == "1.0"
    assert api.legacy_credentials == "auth=abc123hash"
    return api
