"""Tests for pyzm.zm.api -- ZMAPI low-level HTTP wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import requests

from pyzm.models.config import ZMClientConfig
from pyzm.zm.api import ZMAPI


# ===================================================================
# Helpers
# ===================================================================

def _make_config(**overrides) -> ZMClientConfig:
    defaults = dict(
        api_url="https://zm.example.com/zm/api",
        user="admin",
        password="secret",
        verify_ssl=False,
        timeout=10,
    )
    defaults.update(overrides)
    return ZMClientConfig(**defaults)


def _make_mock_auth() -> MagicMock:
    """Create a mock AuthManager."""
    auth = MagicMock()
    auth.api_version = "2.0.0"
    auth.zm_version = "1.36.12"
    auth.auth_enabled = True
    auth.apply_auth.side_effect = lambda url, params: (url, params or {})
    auth.refresh_if_needed.return_value = None
    return auth


# ===================================================================
# TestZMAPI - Construction
# ===================================================================

class TestZMAPIConstruction:
    """Tests for ZMAPI construction and initialization."""

    @patch("pyzm.zm.api.AuthManager")
    @patch("pyzm.zm.api.requests.Session")
    def test_construction_calls_login(self, mock_session_cls, mock_auth_cls):
        mock_auth = _make_mock_auth()
        mock_auth_cls.return_value = mock_auth

        config = _make_config()
        api = ZMAPI(config)

        mock_auth.login.assert_called_once()
        assert api.api_url == "https://zm.example.com/zm/api"

    @patch("pyzm.zm.api.AuthManager")
    @patch("pyzm.zm.api.requests.Session")
    def test_portal_url_derived(self, mock_session_cls, mock_auth_cls):
        mock_auth_cls.return_value = _make_mock_auth()

        config = _make_config()
        api = ZMAPI(config)

        assert api.portal_url == "https://zm.example.com/zm"

    @patch("pyzm.zm.api.AuthManager")
    @patch("pyzm.zm.api.requests.Session")
    def test_properties(self, mock_session_cls, mock_auth_cls):
        mock_auth = _make_mock_auth()
        mock_auth_cls.return_value = mock_auth

        config = _make_config()
        api = ZMAPI(config)

        assert api.api_version == "2.0.0"
        assert api.zm_version == "1.36.12"
        assert api.auth is mock_auth


# ===================================================================
# TestZMAPI - request()
# ===================================================================

class TestZMAPIRequest:
    """Tests for ZMAPI.request() method."""

    def _make_api(self):
        """Create a ZMAPI with mocked internals."""
        with patch("pyzm.zm.api.AuthManager") as mock_auth_cls, \
             patch("pyzm.zm.api.requests.Session") as mock_session_cls:
            mock_auth = _make_mock_auth()
            mock_auth_cls.return_value = mock_auth

            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            config = _make_config()
            api = ZMAPI(config)

            return api, mock_session, mock_auth

    def test_get_request(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"monitors": []}'
        mock_resp.json.return_value = {"monitors": []}
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp

        result = api.request("https://zm.example.com/zm/api/monitors.json")
        assert result == {"monitors": []}
        session.get.assert_called_once()

    def test_post_request(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"success": true}'
        mock_resp.json.return_value = {"success": True}
        mock_resp.raise_for_status.return_value = None
        session.post.return_value = mock_resp

        result = api.request(
            "https://zm.example.com/zm/api/events/1.json",
            method="post",
            payload={"Event[Notes]": "test"},
        )
        assert result == {"success": True}
        session.post.assert_called_once()

    def test_put_request(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"success": true}'
        mock_resp.json.return_value = {"success": True}
        mock_resp.raise_for_status.return_value = None
        session.put.return_value = mock_resp

        result = api.request(
            "https://zm.example.com/zm/api/events/1.json",
            method="put",
            payload={"Event[Notes]": "updated"},
        )
        assert result == {"success": True}
        session.put.assert_called_once()

    def test_delete_request(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = ""
        mock_resp.raise_for_status.return_value = None
        session.delete.return_value = mock_resp

        result = api.request(
            "https://zm.example.com/zm/api/events/1.json",
            method="delete",
        )
        assert result is None
        session.delete.assert_called_once()

    def test_json_response_parsing(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json; charset=utf-8"}
        mock_resp.text = '{"events": [{"Event": {"Id": "1"}}]}'
        mock_resp.json.return_value = {"events": [{"Event": {"Id": "1"}}]}
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp

        result = api.request("https://zm.example.com/zm/api/events.json")
        assert "events" in result
        assert len(result["events"]) == 1

    def test_image_response_handling(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.content = b"\xff\xd8\xff\xe0"
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp

        result = api.request("https://zm.example.com/zm/index.php?view=image&eid=1&fid=1")
        # Should return the raw response for image types
        assert result is mock_resp

    def test_401_retry_logic(self):
        api, session, auth = self._make_api()

        # First call raises 401
        mock_resp_401 = MagicMock()
        mock_resp_401.status_code = 401
        http_error = requests.HTTPError(response=mock_resp_401)
        mock_resp_401.raise_for_status.side_effect = http_error

        # Second call succeeds
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.headers = {"content-type": "application/json"}
        mock_resp_ok.text = '{"ok": true}'
        mock_resp_ok.json.return_value = {"ok": True}
        mock_resp_ok.raise_for_status.return_value = None

        session.get.side_effect = [mock_resp_401, mock_resp_ok]

        result = api.request("https://zm.example.com/zm/api/test.json")
        assert result == {"ok": True}
        auth.handle_401.assert_called_once()

    def test_bad_image_handling_404(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_error = requests.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        session.get.return_value = mock_resp

        with pytest.raises(ValueError, match="BAD_IMAGE"):
            api.request("https://zm.example.com/zm/index.php?view=image&eid=1")

    def test_bad_image_zero_content_length(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html", "content-length": "0"}
        mock_resp.text = ""
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp

        with pytest.raises(ValueError, match="BAD_IMAGE"):
            api.request("https://zm.example.com/zm/test")

    def test_refresh_called_before_request(self):
        api, session, auth = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"ok": true}'
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp

        api.request("https://zm.example.com/zm/api/test.json")
        auth.refresh_if_needed.assert_called()


# ===================================================================
# TestZMAPI - Convenience Methods
# ===================================================================

class TestZMAPIConvenience:
    """Tests for get/post/put/delete convenience methods."""

    def _make_api(self):
        with patch("pyzm.zm.api.AuthManager") as mock_auth_cls, \
             patch("pyzm.zm.api.requests.Session") as mock_session_cls:
            mock_auth = _make_mock_auth()
            mock_auth_cls.return_value = mock_auth

            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            config = _make_config()
            api = ZMAPI(config)
            return api, mock_session

    def test_get_convenience(self):
        api, session = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"test": 1}'
        mock_resp.json.return_value = {"test": 1}
        mock_resp.raise_for_status.return_value = None
        session.get.return_value = mock_resp

        result = api.get("monitors.json")
        assert result == {"test": 1}

    def test_post_convenience(self):
        api, session = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.text = '{"success": true}'
        mock_resp.json.return_value = {"success": True}
        mock_resp.raise_for_status.return_value = None
        session.post.return_value = mock_resp

        result = api.post("events/1.json", data={"key": "val"})
        assert result == {"success": True}

    def test_delete_convenience(self):
        api, session = self._make_api()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = ""
        mock_resp.raise_for_status.return_value = None
        session.delete.return_value = mock_resp

        result = api.delete("events/1.json")
        assert result is None
