"""E2E test fixtures for testing pyzm against a live ZoneMinder instance.

Requires environment variables:
    ZM_API_URL   - Full API URL (e.g. https://zm.local/zm/api)
    ZM_USER      - ZoneMinder username
    ZM_PASSWORD  - ZoneMinder password
    ZM_E2E_WRITE - Set to "1" to enable write-tier tests (optional)
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Override parent autouse fixtures
# ---------------------------------------------------------------------------
# The parent tests/conftest.py defines autouse fixtures `suppress_logger` and
# `no_exit` that mock the logger and builtins.exit(). E2E tests need the real
# logger and real exit() so we override with no-op fixtures at this scope.
# pytest resolves fixtures from the closest conftest first.

@pytest.fixture(autouse=True)
def suppress_logger():
    """No-op override: let real logger run during E2E tests."""
    yield None


@pytest.fixture(autouse=True)
def no_exit():
    """No-op override: let real exit() work during E2E tests."""
    yield None


# ---------------------------------------------------------------------------
# Environment & skip helpers
# ---------------------------------------------------------------------------

def _get_zm_env():
    """Read ZM connection info from environment. Returns dict or None."""
    api_url = os.environ.get("ZM_API_URL")
    user = os.environ.get("ZM_USER")
    password = os.environ.get("ZM_PASSWORD")
    if not all([api_url, user, password]):
        return None
    return {
        "apiurl": api_url,
        "user": user,
        "password": password,
        "disable_ssl_cert_check": True,
    }


def _write_enabled():
    return os.environ.get("ZM_E2E_WRITE") == "1"


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def zm_options_live():
    """ZM options dict from env vars. Skips entire session if unset."""
    env = _get_zm_env()
    if env is None:
        pytest.skip("ZM_API_URL / ZM_USER / ZM_PASSWORD not set")
    return env


@pytest.fixture(scope="session")
def zm_api_live(zm_options_live):
    """Single authenticated ZMApi for the entire test session.

    Token refresh is handled internally by ZMApi._refresh_tokens_if_needed().
    """
    from pyzm.api import ZMApi

    api = ZMApi(options=zm_options_live.copy())
    assert api.authenticated is True, "E2E: login to live ZM failed"
    return api


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zm_api_fresh(zm_options_live):
    """Fresh ZMApi login per test (for auth-specific tests)."""
    from pyzm.api import ZMApi

    return ZMApi(options=zm_options_live.copy())


@pytest.fixture
def e2e_monitor_factory(zm_api_live):
    """Factory that creates monitors with auto-cleanup.

    Usage:
        mon = e2e_monitor_factory(name="test cam", function="Monitor")
        # ... test ...
        # teardown deletes all created monitors
    """
    created = []

    def _create(**kwargs):
        name = kwargs.pop("name", "pyzm_e2e_test_monitor")
        if not name.startswith("pyzm_e2e_test_"):
            name = "pyzm_e2e_test_" + name
        opts = {
            "name": name,
            "function": kwargs.pop("function", "Monitor"),
            "enabled": kwargs.pop("enabled", False),
            "width": kwargs.pop("width", 640),
            "height": kwargs.pop("height", 480),
            "raw": kwargs.pop("raw", {}),
        }
        opts.update(kwargs)
        result = zm_api_live.monitors({"force_reload": True}).add(options=opts)
        # Reload monitors to find the newly created one
        monitors = zm_api_live.monitors({"force_reload": True})
        mon = monitors.find(name=name)
        if mon is not None:
            created.append(mon)
        return mon, result

    yield _create

    # Teardown: delete all monitors we created
    for mon in created:
        try:
            mon.delete()
        except Exception:
            pass


@pytest.fixture
def e2e_config_restorer(zm_api_live):
    """Records original config value and restores it in teardown.

    Usage:
        e2e_config_restorer("ZM_LANG_DEFAULT")
        configs.set(name="ZM_LANG_DEFAULT", val="de_DE")
        # ... test ...
        # teardown restores original value
    """
    saved = []

    def _save(config_name):
        configs = zm_api_live.configs({"force_reload": True})
        original = configs.find(name=config_name)
        saved.append((config_name, original["value"]))

    yield _save

    # Teardown: restore all saved configs
    for config_name, original_value in saved:
        try:
            configs = zm_api_live.configs({"force_reload": True})
            configs.set(name=config_name, val=original_value)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Skip helpers as fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def requires_write():
    """Skip test if ZM_E2E_WRITE is not set."""
    if not _write_enabled():
        pytest.skip("ZM_E2E_WRITE not set to 1")
