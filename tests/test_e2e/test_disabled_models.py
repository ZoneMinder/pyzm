"""E2E: Disabled models are skipped."""

from __future__ import annotations

from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestDisabledModels:

    def test_disabled_model_not_run(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_disabled = mc.model_copy(update={"enabled": False})
        config = DetectorConfig(models=[mc_disabled])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert len(result.detections) == 0

    def test_mixed_enabled_disabled(self):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_enabled = mc.model_copy(update={"enabled": True})
        mc_disabled = mc.model_copy(update={"enabled": False, "name": "disabled_model"})
        config = DetectorConfig(models=[mc_disabled, mc_enabled])
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        for d in result.detections:
            assert d.model_name != "disabled_model"
