"""Tests for pyzm.serve.app -- FastAPI detection server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import FrameStrategy, ServerConfig
from pyzm.models.detection import BBox, Detection, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_detector():
    """Return a mock Detector whose detect() returns a canned result."""
    det = MagicMock()
    det._pipeline = True  # so /health reports models_loaded=True
    det._config = MagicMock()
    det._config.models = [MagicMock()]
    det.detect.return_value = DetectionResult(
        detections=[
            Detection(
                label="person",
                confidence=0.95,
                bbox=BBox(10, 20, 50, 80),
                model_name="yolov4",
            )
        ],
        frame_id="single",
    )
    return det


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a mock Detector.

    The patch must stay active while TestClient is alive so the lifespan
    (which creates the Detector) uses the mock.
    """
    config = ServerConfig(models=["yolov4"])

    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector()
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock()

        from pyzm.serve.app import create_app
        application = create_app(config)

        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


class TestDetect:
    def _make_jpeg(self):
        """Create a minimal valid JPEG bytes payload."""
        import cv2
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    @pytest.mark.integration
    def test_detect_returns_result(self, client):
        jpeg = self._make_jpeg()
        resp = client.post("/detect", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "labels" in data
        assert "boxes" in data
        assert data["labels"] == ["person"]

    @pytest.mark.integration
    def test_detect_with_zones(self, client):
        import json
        jpeg = self._make_jpeg()
        zones_json = json.dumps([{"name": "zone1", "value": [[0, 0], [100, 0], [100, 100], [0, 100]]}])
        resp = client.post(
            "/detect",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
            data={"zones": zones_json},
        )
        assert resp.status_code == 200

    def test_detect_empty_file(self, client):
        resp = client.post("/detect", files={"file": ("test.jpg", b"", "image/jpeg")})
        assert resp.status_code == 400

    @pytest.mark.integration
    def test_detect_bad_image(self, client):
        resp = client.post("/detect", files={"file": ("test.jpg", b"not-a-jpeg", "image/jpeg")})
        assert resp.status_code == 400

    @pytest.mark.integration
    def test_detect_bad_zones_json(self, client):
        jpeg = self._make_jpeg()
        resp = client.post(
            "/detect",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
            data={"zones": "not-json"},
        )
        assert resp.status_code == 400

    @pytest.mark.integration
    def test_detect_no_image_field(self, client):
        """image key is stripped from response."""
        jpeg = self._make_jpeg()
        resp = client.post("/detect", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        assert "image" not in resp.json()


class TestDetectUrls:
    """Tests for the /detect_urls endpoint."""

    def _make_jpeg_bytes(self):
        import cv2
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def test_detect_urls_empty_list(self, client):
        resp = client.post("/detect_urls", json={"urls": []})
        assert resp.status_code == 400

    def test_detect_urls_no_urls_key(self, client):
        resp = client.post("/detect_urls", json={})
        assert resp.status_code == 400

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_fetches_and_detects(self, mock_http, client):
        """Server fetches image from URL and runs detection."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [
                {"frame_id": "snapshot", "url": "http://zm.example.com/image?eid=1&fid=snapshot"},
            ],
            "zm_auth": "token=abc123",
            "verify_ssl": False,
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["labels"] == ["person"]
        assert data["frame_id"] == "snapshot"

        # Verify auth was appended to URL
        fetched_url = mock_http.get.call_args[0][0]
        assert "token=abc123" in fetched_url

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_appends_auth_with_ampersand(self, mock_http, client):
        """zm_auth is appended with & when URL already has query params."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image?eid=1&fid=1"}],
            "zm_auth": "token=xyz",
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        fetched_url = mock_http.get.call_args[0][0]
        assert "&token=xyz" in fetched_url

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_appends_auth_with_question_mark(self, mock_http, client):
        """zm_auth is appended with ? when URL has no query params."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image"}],
            "zm_auth": "token=xyz",
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        fetched_url = mock_http.get.call_args[0][0]
        assert "?token=xyz" in fetched_url

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_no_auth(self, mock_http, client):
        """When zm_auth is empty, URL is not modified."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image?eid=1"}],
            "zm_auth": "",
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        fetched_url = mock_http.get.call_args[0][0]
        assert fetched_url == "http://zm/image?eid=1"

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_all_fetch_failures(self, mock_http, client):
        """When all URL fetches fail, returns empty result."""
        mock_http.get.side_effect = ConnectionError("unreachable")

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image"}],
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["labels"] == []

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_with_zones(self, mock_http, client):
        """Zones are passed through to detector.detect()."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {
            "urls": [{"frame_id": "1", "url": "http://zm/image?eid=1"}],
            "zones": [{"name": "driveway", "value": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200

        # Verify zones were passed to detect()
        det = client.app.state.detector
        call_kwargs = det.detect.call_args[1]
        assert call_kwargs["zones"] is not None
        assert call_kwargs["zones"][0].name == "driveway"

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_picks_best_frame(self, mock_http, client):
        """Multiple frames: best result is returned using frame_strategy."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        det = client.app.state.detector
        det._config.frame_strategy = FrameStrategy.MOST

        # Frame 1: 1 detection, Frame 2: 2 detections
        result1 = DetectionResult(detections=[
            Detection(label="person", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="yolov4"),
        ])
        result2 = DetectionResult(detections=[
            Detection(label="person", confidence=0.9, bbox=BBox(0, 0, 10, 10), model_name="yolov4"),
            Detection(label="car", confidence=0.8, bbox=BBox(20, 20, 60, 60), model_name="yolov4"),
        ])
        det.detect.side_effect = [result1, result2]

        payload = {
            "urls": [
                {"frame_id": "1", "url": "http://zm/image?eid=1&fid=1"},
                {"frame_id": "2", "url": "http://zm/image?eid=1&fid=2"},
            ],
        }
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["labels"]) == 2
        assert data["frame_id"] == "2"

    @pytest.mark.integration
    @patch("pyzm.serve.app.http_requests")
    def test_detect_urls_image_stripped(self, mock_http, client):
        """image key is stripped from response."""
        jpeg_bytes = self._make_jpeg_bytes()
        mock_resp = MagicMock()
        mock_resp.content = jpeg_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_http.get.return_value = mock_resp

        payload = {"urls": [{"frame_id": "1", "url": "http://zm/image"}]}
        resp = client.post("/detect_urls", json=payload)
        assert resp.status_code == 200
        assert "image" not in resp.json()
