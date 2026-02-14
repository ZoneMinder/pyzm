"""E2E: Detector.from_dict with various legacy ml_sequence configs."""

from __future__ import annotations

from pathlib import Path

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


def _resolve(model_name):
    from pyzm.ml.detector import _resolve_model_name
    from pyzm.models.config import Processor
    return _resolve_model_name(model_name, Path(BASE_PATH), Processor.CPU)


class TestFromDict:

    def test_basic_from_dict(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        mc = _resolve(model)
        ml_options = {
            "general": {
                "model_sequence": "object",
                "same_model_sequence_strategy": "first",
            },
            "object": {
                "general": {"pattern": ".*"},
                "sequence": [{
                    "name": model,
                    "object_weights": mc.weights,
                    "object_config": mc.config,
                    "object_labels": mc.labels,
                    "object_framework": mc.framework.value,
                    "object_processor": "cpu",
                }],
            },
        }
        det = Detector.from_dict(ml_options)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)

    def test_from_dict_with_restrictive_pattern(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        mc = _resolve(model)
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {"pattern": "^zzz$"},
                "sequence": [{
                    "object_weights": mc.weights,
                    "object_config": mc.config,
                    "object_labels": mc.labels,
                    "object_framework": mc.framework.value,
                    "object_processor": "cpu",
                }],
            },
        }
        det = Detector.from_dict(ml_options)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_from_dict_match_past_detections(self, tmp_path):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        mc = _resolve(model)
        ml_options = {
            "general": {
                "model_sequence": "object",
                "match_past_detections": "yes",
                "past_det_max_diff_area": "5%",
                "image_path": str(tmp_path),
            },
            "object": {
                "general": {"pattern": ".*"},
                "sequence": [{
                    "object_weights": mc.weights,
                    "object_config": mc.config,
                    "object_labels": mc.labels,
                    "object_framework": mc.framework.value,
                    "object_processor": "cpu",
                }],
            },
        }
        det = Detector.from_dict(ml_options)
        result = det.detect(BIRD_IMAGE)
        first_count = len(result.detections)

        det2 = Detector.from_dict(ml_options)
        result2 = det2.detect(BIRD_IMAGE)
        if first_count > 0:
            assert len(result2.detections) < first_count or len(result2.detections) == 0

    def test_from_dict_disabled_model(self):
        from pyzm.ml.detector import Detector
        model = find_one_model()
        mc = _resolve(model)
        ml_options = {
            "general": {"model_sequence": "object"},
            "object": {
                "general": {"pattern": ".*"},
                "sequence": [{
                    "enabled": "no",
                    "object_weights": mc.weights,
                    "object_config": mc.config,
                    "object_labels": mc.labels,
                    "object_framework": mc.framework.value,
                    "object_processor": "cpu",
                }],
            },
        }
        det = Detector.from_dict(ml_options)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_from_dict_per_label_past_det_area(self, tmp_path):
        from pyzm.models.config import DetectorConfig
        ml_options = {
            "general": {
                "model_sequence": "object",
                "match_past_detections": "yes",
                "past_det_max_diff_area": "5%",
                "car_past_det_max_diff_area": "10%",
                "person_past_det_max_diff_area": "3%",
                "image_path": str(tmp_path),
            },
            "object": {"general": {"pattern": ".*"}, "sequence": []},
        }
        config = DetectorConfig.from_dict(ml_options)
        assert config.past_det_max_diff_area_labels == {"car": "10%", "person": "3%"}

    def test_from_dict_with_aliases(self):
        from pyzm.models.config import DetectorConfig
        ml_options = {
            "general": {
                "model_sequence": "object",
                "aliases": [["car", "bus", "truck"], ["dog", "cat"]],
            },
            "object": {"general": {}, "sequence": []},
        }
        config = DetectorConfig.from_dict(ml_options)
        assert config.aliases == [["car", "bus", "truck"], ["dog", "cat"]]

    def test_from_dict_with_ignore_past_detection_labels(self):
        from pyzm.models.config import DetectorConfig
        ml_options = {
            "general": {
                "model_sequence": "object",
                "ignore_past_detection_labels": ["dog", "cat"],
            },
            "object": {"general": {}, "sequence": []},
        }
        config = DetectorConfig.from_dict(ml_options)
        assert config.ignore_past_detection_labels == ["dog", "cat"]

    def test_from_dict_gateway_settings(self):
        from pyzm.ml.detector import Detector
        ml_options = {
            "general": {
                "model_sequence": "object",
                "ml_gateway": "http://gpu-box:5000",
                "ml_gateway_mode": "url",
                "ml_user": "admin",
                "ml_password": "secret",
            },
            "object": {"general": {}, "sequence": []},
        }
        det = Detector.from_dict(ml_options)
        assert det._gateway == "http://gpu-box:5000"
        assert det._gateway_mode == "url"
        assert det._gateway_username == "admin"
        assert det._gateway_password == "secret"
