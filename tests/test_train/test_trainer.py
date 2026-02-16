"""Tests for pyzm.train.trainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyzm.train.trainer import HardwareInfo, TrainProgress, TrainResult, YOLOTrainer


# ---------------------------------------------------------------------------
# HardwareInfo
# ---------------------------------------------------------------------------

class TestHardwareInfo:
    def test_gpu_display(self):
        hw = HardwareInfo(
            device="cuda:0", gpu_name="NVIDIA GTX 1050 Ti",
            vram_gb=4.0, suggested_batch=16,
        )
        assert "NVIDIA GTX 1050 Ti" in hw.display
        assert "4.0GB" in hw.display

    def test_cpu_display(self):
        hw = HardwareInfo(
            device="cpu", gpu_name=None, vram_gb=0.0, suggested_batch=4,
        )
        assert hw.display == "CPU"


# ---------------------------------------------------------------------------
# TrainProgress
# ---------------------------------------------------------------------------

class TestTrainProgress:
    def test_defaults(self):
        p = TrainProgress()
        assert p.epoch == 0
        assert p.finished is False
        assert p.error is None


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

class TestTrainResult:
    def test_defaults(self):
        r = TrainResult()
        assert r.best_model is None
        assert r.final_mAP50 == 0.0
        assert r.model_size_mb == 0.0


# ---------------------------------------------------------------------------
# YOLOTrainer
# ---------------------------------------------------------------------------

class TestYOLOTrainer:
    def test_init(self, tmp_path: Path):
        trainer = YOLOTrainer(
            base_model="yolo11s",
            project_dir=tmp_path / "proj",
            device="cpu",
        )
        assert trainer.base_model == "yolo11s"
        assert trainer.device == "cpu"

    def test_detect_hardware_cpu_fallback(self):
        """When torch is not available or no CUDA, falls back to CPU."""
        with patch.dict("sys.modules", {"torch": None}):
            hw = YOLOTrainer.detect_hardware()
            assert hw.device == "cpu"
            assert hw.gpu_name is None
            assert hw.suggested_batch == 4

    def test_detect_hardware_with_cuda(self):
        """Mock torch.cuda to test GPU detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.name = "NVIDIA Test GPU"
        mock_props.total_memory = 8 * (1024 ** 3)  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from pyzm.train.trainer import YOLOTrainer as T
            hw = T.detect_hardware()
            assert hw.device == "cuda:0"
            assert hw.gpu_name == "NVIDIA Test GPU"
            assert hw.vram_gb == pytest.approx(8.0)
            assert hw.suggested_batch >= 4

    def test_load_model_adds_pt_extension(self, tmp_path: Path):
        """A plain model name like 'yolo11s' gets .pt appended."""
        trainer = YOLOTrainer("yolo11s", tmp_path)
        mock_yolo = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            trainer._load_model()

        mock_yolo.assert_called_once_with("yolo11s.pt")

    def test_export_onnx_no_model_raises(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        with pytest.raises(FileNotFoundError, match="No best.pt"):
            trainer.export_onnx()

    def test_evaluate_no_model_raises(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        with pytest.raises(FileNotFoundError, match="No trained model"):
            trainer.evaluate("dummy.jpg")

    def test_request_stop(self, tmp_path: Path):
        trainer = YOLOTrainer("yolo11s", tmp_path)
        assert not trainer._stop_event.is_set()
        trainer.request_stop()
        assert trainer._stop_event.is_set()

    def test_export_onnx_copies_to_output_dir(self, tmp_path: Path):
        """When best.pt exists, export should work (mocked)."""
        proj = tmp_path / "proj"
        weights_dir = proj / "runs" / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake model data")

        trainer = YOLOTrainer("yolo11s", proj)
        dest_path = tmp_path / "deploy" / "my_model.onnx"

        mock_model = MagicMock()
        onnx_out = weights_dir / "best.onnx"
        onnx_out.write_bytes(b"fake onnx data")
        mock_model.export.return_value = str(onnx_out)

        mock_yolo = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            result = trainer.export_onnx(output_path=dest_path)

        assert result == dest_path
        assert result.exists()

    def test_evaluate_with_trained_model(self, tmp_path: Path):
        """Test evaluate loads best.pt and parses results."""
        proj = tmp_path / "proj"
        weights_dir = proj / "runs" / "train" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake")

        trainer = YOLOTrainer("yolo11s", proj)

        # Mock YOLO model and its output
        mock_box = MagicMock()
        mock_box.xyxy.__getitem__ = lambda self, idx: MagicMock(
            tolist=lambda: [10.0, 20.0, 100.0, 200.0]
        )
        mock_box.cls.__getitem__ = lambda self, idx: 0
        mock_box.conf.__getitem__ = lambda self, idx: 0.95

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        mock_yolo = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo)}):
            dets = trainer.evaluate("test.jpg")

        assert len(dets) == 1
        assert dets[0]["label"] == "person"
        assert dets[0]["confidence"] == pytest.approx(0.95)
