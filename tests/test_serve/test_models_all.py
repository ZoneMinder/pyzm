"""Tests for --models all and the /models endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import ServerConfig
from pyzm.models.detection import BBox, Detection, DetectionResult


# ---------------------------------------------------------------------------
# ServerConfig validation
# ---------------------------------------------------------------------------

class TestServerConfigModelsAll:
    def test_all_alone_is_valid(self):
        config = ServerConfig(models=["all"])
        assert config.models == ["all"]

    def test_all_mixed_with_other_raises(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            ServerConfig(models=["all", "yolov4"])

    def test_normal_models_still_work(self):
        config = ServerConfig(models=["yolov4", "yolov7"])
        assert config.models == ["yolov4", "yolov7"]


# ---------------------------------------------------------------------------
# /models endpoint
# ---------------------------------------------------------------------------

def _mock_detector(lazy: bool = False):
    """Return a mock Detector with a mock pipeline."""
    det = MagicMock()
    det._pipeline = MagicMock()
    det._config = MagicMock()
    det._config.models = [MagicMock()]
    det.detect.return_value = DetectionResult(
        detections=[
            Detection(
                label="person", confidence=0.95,
                bbox=BBox(10, 20, 50, 80), model_name="yolov4",
            )
        ],
        frame_id="single",
    )

    # Mock the pipeline's _backends list
    mc_mock = MagicMock()
    mc_mock.name = "yolov4"
    mc_mock.type.value = "object"
    mc_mock.framework.value = "opencv"

    backend_mock = MagicMock()
    backend_mock.is_loaded = not lazy

    det._pipeline._backends = [(mc_mock, backend_mock)]
    return det


@pytest.fixture
def client_eager():
    """TestClient with normal eager loading."""
    config = ServerConfig(models=["yolov4"])
    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector(lazy=False)
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

        from pyzm.serve.app import create_app
        application = create_app(config)

        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


@pytest.fixture
def client_all():
    """TestClient simulating --models all (lazy mode)."""
    config = ServerConfig(models=["all"])
    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector(lazy=True)
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

        from pyzm.serve.app import create_app
        application = create_app(config)

        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


class TestModelsEndpoint:
    def test_models_returns_list(self, client_eager):
        resp = client_eager.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "yolov4"
        assert data["models"][0]["loaded"] is True

    def test_models_lazy_shows_not_loaded(self, client_all):
        resp = client_all.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["loaded"] is False


class TestModelsAllLifespan:
    def test_all_creates_detector_with_none_models(self):
        """When config.models == ['all'], Detector is created with models=None."""
        config = ServerConfig(models=["all"])
        with patch("pyzm.serve.app.Detector") as MockDetector:
            mock_det = _mock_detector(lazy=True)
            MockDetector.return_value = mock_det
            mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient

            application = create_app(config)
            with TestClient(application):
                MockDetector.assert_called_once_with(
                    models=None,
                    base_path=config.base_path,
                    processor=config.processor,
                )
                mock_det._ensure_pipeline.assert_called_once_with(lazy=True)

    def test_normal_creates_detector_with_models(self):
        """When config.models is normal, Detector is created with model names."""
        config = ServerConfig(models=["yolov4"])
        with patch("pyzm.serve.app.Detector") as MockDetector:
            mock_det = _mock_detector(lazy=False)
            MockDetector.return_value = mock_det
            mock_det._ensure_pipeline = MagicMock(return_value=mock_det._pipeline)

            from pyzm.serve.app import create_app
            from fastapi.testclient import TestClient

            application = create_app(config)
            with TestClient(application):
                MockDetector.assert_called_once_with(
                    models=["yolov4"],
                    base_path=config.base_path,
                    processor=config.processor,
                )
                mock_det._ensure_pipeline.assert_called_once_with(lazy=False)
