"""E2E: match_past_detections -- dedup, aliases, ignore_labels, per-label overrides."""

from __future__ import annotations

import pickle
from pathlib import Path

from tests.test_e2e.conftest import BIRD_IMAGE, BASE_PATH, det, find_one_model


class TestMatchPastDetections:

    def test_first_run_no_past_file(self, tmp_path):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(
            models=[mc],
            match_past_detections=True,
            image_path=str(tmp_path),
        )
        d = Detector(config=config)
        result = d.detect(BIRD_IMAGE)
        first_count = len(result.detections)
        pkl = tmp_path / "past_detections.pkl"
        if first_count > 0:
            assert pkl.exists()

    def test_second_run_filters_duplicates(self, tmp_path):
        from pyzm.models.config import DetectorConfig, Processor
        from pyzm.ml.detector import Detector, _resolve_model_name
        model = find_one_model()
        mc = _resolve_model_name(model, Path(BASE_PATH), Processor.CPU)
        config = DetectorConfig(
            models=[mc],
            match_past_detections=True,
            past_det_max_diff_area="10%",
            image_path=str(tmp_path),
        )
        d1 = Detector(config=config)
        result1 = d1.detect(BIRD_IMAGE)
        first_count = len(result1.detections)

        d2 = Detector(config=config)
        result2 = d2.detect(BIRD_IMAGE)
        if first_count > 0:
            assert len(result2.detections) < first_count or len(result2.detections) == 0

    def test_aliases_match_cross_label(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections
        past_file = str(tmp_path / "past.pkl")
        with open(past_file, "wb") as fh:
            pickle.dump([[100, 100, 200, 200]], fh)
            pickle.dump(["bus"], fh)

        dets = [det("truck", 100, 100, 200, 200)]
        result = filter_past_detections(
            dets, past_file, "5%",
            aliases=[["car", "bus", "truck"]],
        )
        assert len(result) == 0

    def test_ignore_labels_always_kept(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections
        past_file = str(tmp_path / "past.pkl")
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["person"], fh)

        dets = [det("person", 10, 10, 50, 50)]
        result = filter_past_detections(
            dets, past_file, "5%",
            ignore_labels=["person"],
        )
        assert len(result) == 1

    def test_per_label_area_override(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections
        past_file = str(tmp_path / "past.pkl")
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["car"], fh)

        dets = [det("car", 11, 11, 51, 51)]
        result = filter_past_detections(
            dets, past_file, "0px",
            label_area_overrides={"car": "50%"},
        )
        assert len(result) == 0

    def test_moved_object_passes(self, tmp_path):
        from pyzm.ml.filters import filter_past_detections
        past_file = str(tmp_path / "past.pkl")
        with open(past_file, "wb") as fh:
            pickle.dump([[10, 10, 50, 50]], fh)
            pickle.dump(["person"], fh)

        dets = [det("person", 500, 500, 600, 600)]
        result = filter_past_detections(dets, past_file, "5%")
        assert len(result) == 1
