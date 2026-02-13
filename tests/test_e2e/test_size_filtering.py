"""E2E: max_detection_size filtering at per-model and global level."""

from __future__ import annotations

from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestSizeFiltering:

    def test_per_model_size_filter_percentage(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_small = mc.model_copy(update={"max_detection_size": "0.01%"})
        config = DetectorConfig(models=[mc_small])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_per_model_size_filter_pixels(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_tiny = mc.model_copy(update={"max_detection_size": "1px"})
        config = DetectorConfig(models=[mc_tiny])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_per_model_size_none_allows_all(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_none = mc.model_copy(update={"max_detection_size": None})
        config = DetectorConfig(models=[mc_none])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)

    def test_global_size_filter(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc], max_detection_size="0.01%")
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_global_size_large_allows_all(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(models=[mc], max_detection_size="100%")
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)
