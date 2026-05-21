"""Tests for ZMApi._make_request — retry logic, error handling, content types."""

import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests
import responses

from pyzm.api import ZMApi


@pytest.mark.unit
class TestMakeRequestHTTPMethods:
    """Verify each HTTP method dispatches correctly."""

    @responses.activate
    def test_make_request_get(self, zm_api):
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/test.json",
            json={"ok": True},
            status=200,
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json"
        )

        assert result == {"ok": True}
        assert responses.calls[0].request.method == "GET"

    @responses.activate
    def test_make_request_post(self, zm_api):
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/test.json",
            json={"created": True},
            status=200,
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json",
            payload={"key": "val"},
            type="post",
        )

        assert result == {"created": True}
        assert responses.calls[0].request.method == "POST"

    @responses.activate
    def test_make_request_put(self, zm_api):
        responses.add(
            responses.PUT,
            "https://zm.example.com/zm/api/test.json",
            json={"updated": True},
            status=200,
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json",
            payload={"key": "val"},
            type="put",
        )

        assert result == {"updated": True}
        assert responses.calls[0].request.method == "PUT"

    @responses.activate
    def test_make_request_delete(self, zm_api):
        responses.add(
            responses.DELETE,
            "https://zm.example.com/zm/api/test.json",
            body=b"",
            status=200,
            content_type="application/json",
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json",
            type="delete",
        )

        assert result is None
        assert responses.calls[0].request.method == "DELETE"

    def test_make_request_invalid_type_returns_none(self, zm_api):
        """Invalid type logs error but ValueError is swallowed by the handler."""
        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json",
            type="patch",
        )
        assert result is None


@pytest.mark.unit
class TestMakeRequestReauth:
    """401 and RELOGIN retry behavior."""

    @responses.activate
    def test_make_request_401_triggers_relogin_retry(
        self, zm_api, login_success_response
    ):
        """401 triggers _relogin then retries the request once."""
        # Ensure refresh token is valid for relogin
        zm_api.refresh_token_datetime = (
            datetime.datetime.now() + datetime.timedelta(hours=24)
        )

        # First request: 401
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/test.json",
            json={"error": "Unauthorized"},
            status=401,
        )
        # Relogin
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )
        # Retry: success
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/test.json",
            json={"ok": True},
            status=200,
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json"
        )

        assert result == {"ok": True}
        # 3 calls: original GET, login POST, retry GET
        assert len(responses.calls) == 3

    @responses.activate
    def test_make_request_401_no_retry_when_reauth_false(self, zm_api):
        """401 with reauth=False does not retry."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/test.json",
            json={"error": "Unauthorized"},
            status=401,
        )

        # Should not raise (the code silently returns None for unhandled 401 w/ reauth=False)
        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/test.json",
            reauth=False,
        )

        assert len(responses.calls) == 1


@pytest.mark.unit
class TestMakeRequestContentTypes:
    """Response content-type handling."""

    @responses.activate
    def test_make_request_json_response(self, zm_api):
        """application/json response is parsed to dict."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/data.json",
            json={"data": [1, 2, 3]},
            status=200,
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/data.json"
        )

        assert result == {"data": [1, 2, 3]}

    @responses.activate
    def test_make_request_image_response(self, zm_api):
        """image/* content-type returns the raw response object."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/image.jpg",
            body=b"\xff\xd8\xff\xe0",
            status=200,
            content_type="image/jpeg",
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/image.jpg"
        )

        # Returns response object, not parsed JSON
        assert hasattr(result, "status_code")
        assert result.status_code == 200

    @responses.activate
    def test_make_request_404_raises_bad_image(self, zm_api):
        """404 raises ValueError('BAD_IMAGE')."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/image.jpg",
            json={"error": "Not found"},
            status=404,
        )

        with pytest.raises(ValueError, match="BAD_IMAGE"):
            zm_api._make_request(
                url="https://zm.example.com/zm/api/image.jpg"
            )

    @responses.activate
    def test_make_request_relogin_value_error(
        self, zm_api, login_success_response
    ):
        """Non-JSON, non-image response with content triggers RELOGIN retry."""
        zm_api.refresh_token_datetime = (
            datetime.datetime.now() + datetime.timedelta(hours=24)
        )

        # First: returns HTML (not JSON, not image, has content)
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/resource.json",
            body=b"<html>Login required</html>",
            status=200,
            content_type="text/html",
            headers={"content-length": "27"},
        )
        # Relogin
        responses.add(
            responses.POST,
            "https://zm.example.com/zm/api/host/login.json",
            json=login_success_response,
            status=200,
        )
        # Retry: proper JSON
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/resource.json",
            json={"data": "ok"},
            status=200,
        )

        result = zm_api._make_request(
            url="https://zm.example.com/zm/api/resource.json"
        )

        assert result == {"data": "ok"}
        assert len(responses.calls) == 3


@pytest.mark.unit
class TestMakeRequestTokenInjection:
    """Verify tokens/credentials are injected into requests."""

    @responses.activate
    def test_jwt_token_added_to_query(self, zm_api):
        """JWT access token is added as query parameter."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/test.json",
            json={"ok": True},
            status=200,
        )

        zm_api._make_request(url="https://zm.example.com/zm/api/test.json")

        request_url = responses.calls[0].request.url
        assert "token=test_access_token_abc123" in request_url

    @responses.activate
    def test_legacy_credentials_appended_to_url(self, zm_api_legacy):
        """Legacy credentials are appended to URL."""
        responses.add(
            responses.GET,
            "https://zm.example.com/zm/api/test.json",
            json={"ok": True},
            status=200,
            match_querystring=False,
        )

        zm_api_legacy._make_request(
            url="https://zm.example.com/zm/api/test.json"
        )

        request_url = responses.calls[0].request.url
        assert "auth=abc123hash" in request_url
