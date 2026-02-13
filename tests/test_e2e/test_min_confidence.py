"""E2E: min_confidence threshold per-model filtering."""

from __future__ import annotations

from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestMinConfidence:

    def test_high_confidence_threshold_filters(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_high = mc.model_copy(update={"min_confidence": 0.99})
        config = DetectorConfig(models=[mc_high])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        for d in result.detections:
            assert d.confidence >= 0.99

    def test_low_confidence_threshold_allows_more(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)

        mc_low = mc.model_copy(update={"min_confidence": 0.01})
        config_low = DetectorConfig(models=[mc_low])
        det_low = Detector(config=config_low)
        result_low = det_low.detect(BIRD_IMAGE)

        mc_high = mc.model_copy(update={"min_confidence": 0.9})
        config_high = DetectorConfig(models=[mc_high])
        det_high = Detector(config=config_high)
        result_high = det_high.detect(BIRD_IMAGE)

        assert len(result_low.detections) >= len(result_high.detections)
