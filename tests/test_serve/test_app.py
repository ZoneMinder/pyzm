"""Tests for pyzm.serve.app -- FastAPI detection server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import ServerConfig
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
