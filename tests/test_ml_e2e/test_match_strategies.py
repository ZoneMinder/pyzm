"""E2E: Match strategies -- FIRST, MOST, MOST_UNIQUE, UNION."""

from __future__ import annotations

from pathlib import Path

from tests.test_ml_e2e.conftest import BIRD_IMAGE, BASE_PATH, find_one_model


class TestMatchStrategies:

    def test_first_strategy(self):
        from pyzm.models.config import DetectorConfig, MatchStrategy, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc1 = mc.model_copy(update={"name": "first_model", "pattern": ".*"})
        mc2 = mc.model_copy(update={"name": "second_model", "pattern": ".*"})
        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.FIRST,
        )
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        if result.matched:
            model_names = {d.model_name for d in result.detections}
            assert "first_model" in model_names
            assert "second_model" not in model_names

    def test_most_strategy(self):
        from pyzm.models.config import DetectorConfig, MatchStrategy, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc_few = mc.model_copy(update={"name": "few_model", "pattern": "^zzz$"})
        mc_many = mc.model_copy(update={"name": "many_model", "pattern": ".*"})
        config = DetectorConfig(
            models=[mc_few, mc_many],
            match_strategy=MatchStrategy.MOST,
        )
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        if result.matched:
            model_names = {d.model_name for d in result.detections}
            assert "many_model" in model_names

    def test_union_strategy(self):
        from pyzm.models.config import DetectorConfig, MatchStrategy, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc1 = mc.model_copy(update={"name": "union_a", "pattern": ".*"})
        mc2 = mc.model_copy(update={"name": "union_b", "pattern": ".*"})
        config = DetectorConfig(
            models=[mc1, mc2],
            match_strategy=MatchStrategy.UNION,
        )
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        if result.matched:
            model_names = {d.model_name for d in result.detections}
            assert "union_a" in model_names
            assert "union_b" in model_names

    def test_most_unique_strategy(self):
        from pyzm.models.config import DetectorConfig, MatchStrategy, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        mc1 = mc.model_copy(update={"name": "unique_model", "pattern": ".*"})
        config = DetectorConfig(
            models=[mc1],
            match_strategy=MatchStrategy.MOST_UNIQUE,
        )
        det = Detector(config=config)
        result = det.detect(BIRD_IMAGE)
        assert isinstance(result.detections, list)
