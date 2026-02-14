"""E2E tests for ZoneMinder authentication and version info."""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.zm_e2e


class TestAuth:
    def test_login_succeeds(self, zm_client_fresh):
        """A fresh login should populate version info without raising."""
        assert zm_client_fresh.zm_version is not None
        assert zm_client_fresh.api_version is not None

    def test_zm_version_format(self, zm_client_fresh):
        """zm_version should be a dotted numeric string (e.g. '1.36.33')."""
        assert re.match(r"\d+\.\d+", zm_client_fresh.zm_version)

    def test_api_version_format(self, zm_client_fresh):
        """api_version should be a dotted numeric string (e.g. '2.0')."""
        assert re.match(r"\d+\.\d+", zm_client_fresh.api_version)

    def test_session_client_still_works(self, zm_client):
        """The session-scoped client should still be functional (token not expired)."""
        monitors = zm_client.monitors()
        assert isinstance(monitors, list)
