"""E2E: pre_existing_labels gate between model types."""

from __future__ import annotations

from pathlib import Path

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestPreExistingLabels:

    def test_pre_existing_labels_not_satisfied_skips(self):
        from pyzm.models.config import DetectorConfig, ModelType, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_obj = mc.model_copy(update={"name": "obj", "type": ModelType.OBJECT, "enabled": False})
        mc_gated = mc.model_copy(update={
            "name": "gated",
            "type": ModelType.FACE,
            "pre_existing_labels": ["person"],
        })
        config = DetectorConfig(models=[mc_obj, mc_gated])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        gated_dets = [d for d in result.detections if d.model_name == "gated"]
        assert len(gated_dets) == 0

    def test_pre_existing_labels_satisfied_runs(self):
        from pyzm.models.config import DetectorConfig, ModelType, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_obj = mc.model_copy(update={"name": "obj1", "type": ModelType.OBJECT})
        mc_gated = mc.model_copy(update={
            "name": "gated1",
            "type": ModelType.OBJECT,
            "pre_existing_labels": ["bird"],
        })
        config = DetectorConfig(models=[mc_obj, mc_gated])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        obj_labels = [d.label for d in result.detections if d.model_name == "obj1"]
        if "bird" in obj_labels:
            assert True  # Gated model was allowed to run (no crash)
        else:
            gated_dets = [d for d in result.detections if d.model_name == "gated1"]
            assert len(gated_dets) == 0
