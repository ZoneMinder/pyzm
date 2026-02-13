"""E2E: model_width / model_height overrides."""

from __future__ import annotations

from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestModelDimensions:

    def test_custom_dimensions(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_custom = mc.model_copy(update={"model_width": 320, "model_height": 320})
        config = DetectorConfig(models=[mc_custom])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)

    def test_default_dimensions(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        assert mc.model_width is None
        assert mc.model_height is None
        config = DetectorConfig(models=[mc])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)
