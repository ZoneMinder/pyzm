"""Tests for ModelPipeline.prepare() -- lazy backend creation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pyzm.models.config import (
    DetectorConfig,
    MatchStrategy,
    ModelConfig,
    ModelFramework,
    ModelType,
)
from pyzm.models.detection import BBox, Detection


def _det(label: str, conf: float = 0.9, model: str = "test") -> Detection:
    return Detection(
        label=label, confidence=conf,
        bbox=BBox(x1=10, y1=10, x2=50, y2=50), model_name=model,
    )


def _make_mc(name: str = "test") -> ModelConfig:
    return ModelConfig(name=name, type=ModelType.OBJECT, framework=ModelFramework.OPENCV)


class TestPrepare:
    """Tests for ModelPipeline.prepare() (lazy loading)."""

    @patch("pyzm.ml.pipeline._create_backend")
    def test_prepare_creates_backends_without_loading(self, mock_create):
        """prepare() should call _create_backend but NOT backend.load()."""
        backend = MagicMock()
        backend.name = "test"
        mock_create.return_value = backend

        config = DetectorConfig(models=[_make_mc()])

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.prepare()

        mock_create.assert_called_once()
        backend.load.assert_not_called()
        assert len(pipeline._backends) == 1
        assert pipeline._loaded is True

    @patch("pyzm.ml.pipeline._create_backend")
    def test_prepare_skips_disabled_models(self, mock_create):
        mc_on = _make_mc("enabled")
        mc_off = ModelConfig(name="disabled", enabled=False, framework=ModelFramework.OPENCV)

        backend = MagicMock()
        backend.name = "enabled"
        mock_create.return_value = backend

        config = DetectorConfig(models=[mc_on, mc_off])

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.prepare()

        assert mock_create.call_count == 1

    @patch("pyzm.ml.pipeline._create_backend")
    def test_prepare_is_idempotent(self, mock_create):
        """Calling prepare() twice should not create backends twice."""
        backend = MagicMock()
        backend.name = "test"
        mock_create.return_value = backend

        config = DetectorConfig(models=[_make_mc()])

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.prepare()
        pipeline.prepare()

        assert mock_create.call_count == 1

    @patch("pyzm.ml.pipeline._create_backend")
    def test_load_is_idempotent(self, mock_create):
        """Calling load() twice should not create backends twice."""
        backend = MagicMock()
        backend.name = "test"
        mock_create.return_value = backend

        config = DetectorConfig(models=[_make_mc()])

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.load()
        pipeline.load()

        assert mock_create.call_count == 1

    @patch("pyzm.ml.pipeline._create_backend")
    def test_load_after_prepare_is_noop(self, mock_create):
        """load() after prepare() should not reload backends."""
        backend = MagicMock()
        backend.name = "test"
        mock_create.return_value = backend

        config = DetectorConfig(models=[_make_mc()])

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.prepare()
        pipeline.load()  # Should be a no-op

        # _create_backend called once (by prepare), load() never called
        assert mock_create.call_count == 1
        backend.load.assert_not_called()

    @patch("pyzm.ml.pipeline._create_backend")
    @patch("pyzm.ml.pipeline.filter_by_zone")
    def test_lazy_backend_loads_on_first_detect(self, mock_zone_filter, mock_create):
        """After prepare(), running pipeline triggers backend.detect()
        which should cause lazy loading (handled by backend internals)."""
        backend = MagicMock()
        backend.name = "test"
        backend.detect.return_value = [_det("person")]
        mock_create.return_value = backend
        mock_zone_filter.side_effect = lambda dets, zones, shape: (dets, [])

        config = DetectorConfig(models=[_make_mc()], match_strategy=MatchStrategy.FIRST)

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.prepare()

        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)

        result = pipeline.run(mock_image)

        # backend.detect() was called (lazy load happens inside)
        backend.detect.assert_called_once()
        assert len(result.detections) == 1

    @patch("pyzm.ml.pipeline._create_backend")
    def test_prepare_handles_backend_creation_error(self, mock_create):
        """If _create_backend raises, prepare() should log and continue."""
        mock_create.side_effect = ImportError("No coral module")

        config = DetectorConfig(models=[_make_mc()])

        from pyzm.ml.pipeline import ModelPipeline
        pipeline = ModelPipeline(config)
        pipeline.prepare()

        assert len(pipeline._backends) == 0
        assert pipeline._loaded is True
