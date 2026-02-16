"""Tests for the headless training pipeline."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyzm.train.dataset import YOLODataset
from pyzm.train.local_import import import_local_dataset, validate_yolo_folder
from pyzm.train.trainer import HardwareInfo, TrainResult
from pyzm.train.verification import VerificationStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_dataset(tmp_path: Path, class_names: list[str] | None = None) -> Path:
    """Create a minimal YOLO dataset folder on disk."""
    class_names = class_names or ["cat", "dog"]
    ds_dir = tmp_path / "my_dataset"
    images_dir = ds_dir / "images"
    labels_dir = ds_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # data.yaml
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (ds_dir / "data.yaml").write_text(f"names:\n{names_block}\n")

    # Two tiny images (1x1 white PNG bytes)
    _1x1_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    for name in ("img_001.png", "img_002.png"):
        (images_dir / name).write_bytes(_1x1_png)

    # Matching labels
    (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    (labels_dir / "img_002.txt").write_text("1 0.2 0.8 0.4 0.4\n")

    return ds_dir


# ---------------------------------------------------------------------------
# validate_yolo_folder
# ---------------------------------------------------------------------------

class TestValidateYOLOFolder:
    def test_valid_standard_layout(self, tmp_path: Path) -> None:
        ds_dir = _make_yolo_dataset(tmp_path)
        result = validate_yolo_folder(ds_dir)
        assert isinstance(result, dict)
        assert "names" in result
        assert "_splits" in result
        assert len(result["_splits"]) == 1

    def test_missing_data_yaml(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = validate_yolo_folder(empty)
        assert isinstance(result, str)
        assert "Missing data.yaml" in result

    def test_not_a_directory(self, tmp_path: Path) -> None:
        fake = tmp_path / "not_a_dir"
        result = validate_yolo_folder(fake)
        assert isinstance(result, str)
        assert "Not a directory" in result

    def test_names_as_list(self, tmp_path: Path) -> None:
        ds_dir = _make_yolo_dataset(tmp_path)
        (ds_dir / "data.yaml").write_text("names:\n  - cat\n  - dog\n")
        result = validate_yolo_folder(ds_dir)
        assert isinstance(result, dict)
        assert result["names"] == {0: "cat", 1: "dog"}


# ---------------------------------------------------------------------------
# import_local_dataset (no Streamlit)
# ---------------------------------------------------------------------------

class TestImportLocalDataset:
    def test_basic_import(self, tmp_path: Path) -> None:
        ds_dir = _make_yolo_dataset(tmp_path)
        result = validate_yolo_folder(ds_dir)
        assert isinstance(result, dict)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        ds = YOLODataset(project_dir, classes=["cat", "dog"])
        ds.init_project()
        store = VerificationStore(project_dir)

        img_count, det_count = import_local_dataset(
            ds, store, result["_splits"], result["names"],
        )
        assert img_count == 2
        assert det_count == 2

    def test_progress_callback(self, tmp_path: Path) -> None:
        ds_dir = _make_yolo_dataset(tmp_path)
        result = validate_yolo_folder(ds_dir)
        assert isinstance(result, dict)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        ds = YOLODataset(project_dir, classes=["cat", "dog"])
        ds.init_project()
        store = VerificationStore(project_dir)

        calls: list[tuple[int, int]] = []

        def cb(current: int, total: int) -> None:
            calls.append((current, total))

        import_local_dataset(
            ds, store, result["_splits"], result["names"],
            progress_callback=cb,
        )
        assert len(calls) == 2
        assert calls[-1] == (2, 2)

    def test_max_images(self, tmp_path: Path) -> None:
        ds_dir = _make_yolo_dataset(tmp_path)
        result = validate_yolo_folder(ds_dir)
        assert isinstance(result, dict)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        ds = YOLODataset(project_dir, classes=["cat", "dog"])
        ds.init_project()
        store = VerificationStore(project_dir)

        img_count, _ = import_local_dataset(
            ds, store, result["_splits"], result["names"], max_images=1,
        )
        assert img_count == 1


# ---------------------------------------------------------------------------
# run_pipeline (mocked trainer)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @patch("pyzm.train.pipeline.YOLOTrainer")
    def test_full_pipeline(self, mock_trainer_cls: MagicMock, tmp_path: Path) -> None:
        from pyzm.train.pipeline import run_pipeline

        ds_dir = _make_yolo_dataset(tmp_path)

        # Mock hardware detection
        mock_trainer_cls.detect_hardware.return_value = HardwareInfo(
            device="cpu", gpu_name=None, vram_gb=0.0, suggested_batch=8,
        )

        # Mock trainer instance
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.train.return_value = TrainResult(
            best_model=tmp_path / "best.pt",
            last_model=tmp_path / "last.pt",
            best_epoch=5,
            final_mAP50=0.85,
            final_mAP50_95=0.60,
            total_epochs=10,
            elapsed_seconds=120.0,
            model_size_mb=22.5,
        )
        mock_trainer.export_onnx.return_value = tmp_path / "best.onnx"

        result = run_pipeline(
            ds_dir,
            workspace_dir=tmp_path / "workspace",
            epochs=10,
            device="cpu",
        )

        assert result.final_mAP50 == 0.85
        assert result.best_epoch == 5
        mock_trainer.train.assert_called_once()
        mock_trainer.export_onnx.assert_called_once()

    @patch("pyzm.train.pipeline.YOLOTrainer")
    def test_no_best_model_skips_export(
        self, mock_trainer_cls: MagicMock, tmp_path: Path,
    ) -> None:
        from pyzm.train.pipeline import run_pipeline

        ds_dir = _make_yolo_dataset(tmp_path)

        mock_trainer_cls.detect_hardware.return_value = HardwareInfo(
            device="cpu", gpu_name=None, vram_gb=0.0, suggested_batch=8,
        )
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.train.return_value = TrainResult(best_model=None)

        result = run_pipeline(
            ds_dir,
            workspace_dir=tmp_path / "workspace",
            epochs=5,
            device="cpu",
        )

        assert result.best_model is None
        mock_trainer.export_onnx.assert_not_called()

    def test_invalid_dataset_raises(self, tmp_path: Path) -> None:
        from pyzm.train.pipeline import run_pipeline

        with pytest.raises(ValueError, match="Not a directory"):
            run_pipeline(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestCLIArgParsing:
    def test_headless_mode_detected(self) -> None:
        """When a dataset path is given, headless mode is selected."""
        from pyzm.train.__main__ import main

        with patch("pyzm.train.__main__._run_headless") as mock_headless, \
             patch("sys.argv", ["prog", "/tmp/ds"]):
            main()
            mock_headless.assert_called_once()
            args = mock_headless.call_args[0][0]
            assert args.dataset == "/tmp/ds"

    def test_ui_mode_detected(self) -> None:
        """When no dataset is given, UI mode is selected."""
        from pyzm.train.__main__ import main

        with patch("pyzm.train.__main__._run_ui") as mock_ui, \
             patch("sys.argv", ["prog"]):
            main()
            mock_ui.assert_called_once()

    def test_headless_flags_parsed(self) -> None:
        from pyzm.train.__main__ import main

        with patch("pyzm.train.__main__._run_headless") as mock_headless, \
             patch("sys.argv", [
                 "prog", "/tmp/ds",
                 "--epochs", "100",
                 "--batch", "8",
                 "--model", "yolo11n",
                 "--imgsz", "320",
                 "--val-ratio", "0.15",
                 "--device", "cuda:0",
                 "--project-name", "myproject",
             ]):
            main()
            args = mock_headless.call_args[0][0]
            assert args.epochs == 100
            assert args.batch == 8
            assert args.model == "yolo11n"
            assert args.imgsz == 320
            assert args.val_ratio == 0.15
            assert args.device == "cuda:0"
            assert args.project_name == "myproject"

    def test_ui_flags_preserved(self) -> None:
        from pyzm.train.__main__ import main

        with patch("pyzm.train.__main__._run_ui") as mock_ui, \
             patch("sys.argv", [
                 "prog",
                 "--host", "127.0.0.1",
                 "--port", "9000",
                 "--processor", "cpu",
             ]):
            main()
            args = mock_ui.call_args[0][0]
            assert args.host == "127.0.0.1"
            assert args.port == 9000
            assert args.processor == "cpu"
