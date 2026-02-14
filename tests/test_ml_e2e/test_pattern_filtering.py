"""E2E: Regex pattern filtering at per-model and global level."""

from __future__ import annotations

from pathlib import Path

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestPatternFiltering:

    def test_per_model_pattern_filters(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_filtered = mc.model_copy(update={"pattern": "^zzz_no_match$"})
        config = DetectorConfig(models=[mc_filtered])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_per_model_pattern_allows_match(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_open = mc.model_copy(update={"pattern": ".*"})
        config = DetectorConfig(models=[mc_open])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)

    def test_global_pattern_filters(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc], pattern="^zzz_no_match$")
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_global_pattern_allows_specific(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc], pattern="bird")
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        for d in result.detections:
            assert d.label == "bird"
