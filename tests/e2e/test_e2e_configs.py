"""E2E tests for Configs against a live ZoneMinder.

Readonly tests verify list/find. Write tests modify config values (with restore).
"""

import pytest

from pyzm.helpers.Configs import Configs

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Readonly tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_readonly
class TestConfigList:
    """Verify configs() returns a valid Configs collection."""

    def test_returns_configs_instance(self, zm_api_live):
        result = zm_api_live.configs()
        assert isinstance(result, Configs)

    def test_list_returns_list(self, zm_api_live):
        configs = zm_api_live.configs()
        lst = configs.list()
        assert isinstance(lst, list)

    def test_list_nonempty(self, zm_api_live):
        configs = zm_api_live.configs()
        assert len(configs.list()) > 0

    def test_item_structure(self, zm_api_live):
        """Each config item should be a dict with {'Config': {'Id', 'Name', 'Value'}}."""
        configs = zm_api_live.configs()
        item = configs.list()[0]
        assert isinstance(item, dict)
        assert "Config" in item
        inner = item["Config"]
        assert "Id" in inner
        assert "Name" in inner
        assert "Value" in inner


@pytest.mark.e2e_readonly
class TestConfigFind:
    """Verify Configs.find() search logic."""

    def test_find_by_name(self, zm_api_live):
        configs = zm_api_live.configs({"force_reload": True})
        result = configs.find(name="ZM_AUTH_TYPE")
        assert result is not None
        assert isinstance(result, dict)
        assert isinstance(result["id"], int)
        assert isinstance(result["name"], str)
        assert isinstance(result["value"], str)
        assert result["name"] == "ZM_AUTH_TYPE"

    def test_find_zm_auth_type(self, zm_api_live):
        """ZM_AUTH_TYPE should exist on any ZM installation."""
        configs = zm_api_live.configs({"force_reload": True})
        result = configs.find(name="ZM_AUTH_TYPE")
        assert result is not None
        # Value should be "builtin" or "remote"
        assert result["value"] in ("builtin", "remote"), \
            f"Unexpected ZM_AUTH_TYPE value: {result['value']}"

    def test_find_nonexistent_returns_none(self, zm_api_live):
        """Configs.find() returns None when no match — consistent with
        Monitors.find() and States.find()."""
        configs = zm_api_live.configs({"force_reload": True})
        result = configs.find(name="ZM_PYZM_E2E_NONEXISTENT_CONFIG_XYZ")
        assert result is None


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e_write
class TestConfigSet:
    """Test modifying config values (with auto-restore)."""

    def test_set_config_value(self, zm_api_live, e2e_config_restorer, requires_write):
        """Modify a safe config value and verify the change."""
        config_name = "ZM_LANG_DEFAULT"
        e2e_config_restorer(config_name)

        configs = zm_api_live.configs({"force_reload": True})
        original = configs.find(name=config_name)
        new_value = "de_DE" if original["value"] != "de_DE" else "en_GB"

        configs.set(name=config_name, val=new_value)

        # Reload and verify
        configs = zm_api_live.configs({"force_reload": True})
        updated = configs.find(name=config_name)
        assert updated["value"] == new_value

