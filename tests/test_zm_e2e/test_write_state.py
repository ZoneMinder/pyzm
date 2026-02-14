"""E2E write tests for ZM state management (requires ZM_E2E_WRITE=1).

These tests call stop/start/restart and wait for ZM to come back up
between each state change.
"""

from __future__ import annotations

import pytest

from tests.test_zm_e2e.conftest import _get, wait_for_zm

pytestmark = [pytest.mark.zm_e2e, pytest.mark.zm_e2e_write]


def _wait(config) -> bool:
    """Wait for ZM to come back up, using credentials from env."""
    return wait_for_zm(
        config.api_url,
        timeout=90,
        verify_ssl=config.verify_ssl,
        user=_get("ZM_USER", "admin"),
        password=_get("ZM_PASSWORD", "admin"),
    )


class TestWriteState:
    def test_stop_and_start(self, zm_e2e_config, zm_client):
        """stop() then start() should cycle ZM. Reuse existing client
        since creating a new one after stop would try to login against
        a stopped server."""
        zm_client.stop()
        zm_client.start()
        assert _wait(zm_e2e_config), "ZM did not come back up after stop() + start()"

    def test_restart(self, zm_e2e_config, zm_client):
        """restart() should cycle ZM and the API should respond after."""
        zm_client.restart()
        assert _wait(zm_e2e_config), "ZM did not come back up after restart()"

    def test_set_state_direct(self, zm_e2e_config, zm_client):
        """set_state('restart') should also work (underlying method)."""
        zm_client.set_state("restart")
        assert _wait(zm_e2e_config), "ZM did not come back up after set_state('restart')"
