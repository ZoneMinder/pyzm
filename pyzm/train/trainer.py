"""Ultralytics YOLO training wrapper.

Handles model loading, hardware detection, training with progress callbacks,
evaluation, and ONNX export.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("pyzm.train")


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    device: str           # "cuda:0", "cpu"
    gpu_name: str | None  # e.g. "NVIDIA GTX 1050 Ti"
    vram_gb: float        # 0.0 for CPU
    suggested_batch: int  # Based on VRAM

    @property
    def display(self) -> str:
        if self.gpu_name:
            return f"GPU: {self.gpu_name} ({self.vram_gb:.1f}GB)"
        return "CPU"


@dataclass
class TrainProgress:
    """Training progress snapshot."""

    epoch: int = 0
    total_epochs: int = 0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    message: str = ""
    finished: bool = False
    error: str | None = None


@dataclass
class TrainResult:
    """Final training results."""

    best_model: Path | None = None
    last_model: Path | None = None
    best_epoch: int = 0
    final_mAP50: float = 0.0
    final_mAP50_95: float = 0.0
    total_epochs: int = 0
    elapsed_seconds: float = 0.0
    model_size_mb: float = 0.0
    log_lines: list[str] = field(default_factory=list)


class YOLOTrainer:
    """Wraps Ultralytics YOLO for fine-tuning.

    Parameters
    ----------
    base_model : str
        Model name (e.g. ``"yolo11s"``) or path to a ``.pt`` file.
        Plain names auto-download from Ultralytics hub.
    project_dir : Path
        Project directory (contains dataset.yaml, runs/).
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda:0"`` etc.
    """

    def __init__(
        self,
        base_model: str,
        project_dir: Path,
        device: str = "auto",
    ) -> None:
        self.base_model = base_model
        self.project_dir = Path(project_dir)
        self.device = device
        self._model: Any = None  # ultralytics.YOLO
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_hardware() -> HardwareInfo:
        """Detect GPU/CPU and suggest training batch size."""
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_mem / (1024 ** 3)
                # Rough heuristic: 2GB per batch of 8 at 640px
                suggested = max(4, min(32, int(vram_gb / 2) * 8))
                return HardwareInfo(
                    device="cuda:0",
                    gpu_name=props.name,
                    vram_gb=vram_gb,
                    suggested_batch=suggested,
                )
        except (ImportError, Exception):
            pass

        return HardwareInfo(
            device="cpu",
            gpu_name=None,
            vram_gb=0.0,
            suggested_batch=4,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the Ultralytics YOLO model."""
        from ultralytics import YOLO

        model_path = self.base_model
        # If it doesn't look like a file path, assume it's a hub name
        if not Path(model_path).suffix:
            model_path = f"{model_path}.pt"
        self._model = YOLO(model_path)
        return self._model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_yaml: Path,
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        progress_callback: Callable[[TrainProgress], None] | None = None,
    ) -> TrainResult:
        """Run fine-tuning.

        This is intended to be called from a background thread.
        Use :meth:`request_stop` to signal early stopping.

        Parameters
        ----------
        dataset_yaml:
            Path to the YOLO dataset.yaml.
        epochs:
            Number of training epochs.
        batch:
            Batch size.
        imgsz:
            Training image size.
        progress_callback:
            Called after each epoch with a :class:`TrainProgress` snapshot.
        """
        import time

        self._stop_event.clear()
        model = self._load_model()

        device = self.device
        if device == "auto":
            hw = self.detect_hardware()
            device = hw.device

        runs_dir = self.project_dir / "runs"

        progress = TrainProgress(total_epochs=epochs)

        def _on_train_epoch_end(trainer: Any) -> None:
            """Ultralytics callback after each training epoch."""
            if self._stop_event.is_set():
                raise KeyboardInterrupt("Training stopped by user")

            progress.epoch = trainer.epoch + 1
            metrics = trainer.metrics or {}
            progress.box_loss = float(trainer.loss_items[0]) if trainer.loss_items is not None else 0.0
            progress.cls_loss = float(trainer.loss_items[1]) if trainer.loss_items is not None and len(trainer.loss_items) > 1 else 0.0
            progress.mAP50 = float(metrics.get("metrics/mAP50(B)", 0.0))
            progress.mAP50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
            progress.message = (
                f"Epoch {progress.epoch}/{epochs} | "
                f"mAP50={progress.mAP50:.3f}"
            )
            if progress_callback:
                progress_callback(progress)

        model.add_callback("on_train_epoch_end", _on_train_epoch_end)

        start = time.monotonic()
        try:
            results = model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                project=str(runs_dir),
                name="train",
                exist_ok=True,
                verbose=False,
            )
        except KeyboardInterrupt:
            logger.info("Training stopped by user")
            progress.finished = True
            progress.message = "Training stopped by user"
            if progress_callback:
                progress_callback(progress)
        except Exception as exc:
            progress.error = str(exc)
            progress.finished = True
            if progress_callback:
                progress_callback(progress)
            raise
        else:
            progress.finished = True
            progress.message = "Training complete"
            if progress_callback:
                progress_callback(progress)

        elapsed = time.monotonic() - start

        # Locate output weights
        weights_dir = runs_dir / "train" / "weights"
        best_pt = weights_dir / "best.pt" if weights_dir.exists() else None
        last_pt = weights_dir / "last.pt" if weights_dir.exists() else None
        if best_pt and not best_pt.exists():
            best_pt = None
        if last_pt and not last_pt.exists():
            last_pt = None

        model_size = best_pt.stat().st_size / (1024 * 1024) if best_pt else 0.0

        return TrainResult(
            best_model=best_pt,
            last_model=last_pt,
            best_epoch=progress.epoch,
            final_mAP50=progress.mAP50,
            final_mAP50_95=progress.mAP50_95,
            total_epochs=progress.epoch,
            elapsed_seconds=elapsed,
            model_size_mb=model_size,
        )

    def request_stop(self) -> None:
        """Signal the training loop to stop after the current epoch."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, image: "Any") -> list[dict]:
        """Run the fine-tuned model on a single image.

        Parameters
        ----------
        image:
            Image path (str) or numpy array.

        Returns
        -------
        List of dicts with keys: label, confidence, bbox (x1,y1,x2,y2).
        """
        if self._model is None:
            # Try loading best.pt from project
            best = self.project_dir / "runs" / "train" / "weights" / "best.pt"
            if best.exists():
                from ultralytics import YOLO
                self._model = YOLO(str(best))
            else:
                raise FileNotFoundError("No trained model found. Run training first.")

        results = self._model(image, verbose=False)
        detections: list[dict] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id] if cls_id in r.names else str(cls_id)
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })
        return detections

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, output_dir: Path | None = None) -> Path:
        """Export the best trained model to ONNX format.

        Parameters
        ----------
        output_dir:
            Where to copy the exported ONNX file.  Defaults to the
            project's ``runs/train/weights/`` directory.

        Returns
        -------
        Path to the exported ``.onnx`` file.
        """
        best_pt = self.project_dir / "runs" / "train" / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"No best.pt found at {best_pt}")

        from ultralytics import YOLO

        model = YOLO(str(best_pt))
        onnx_path = model.export(format="onnx")
        onnx_path = Path(onnx_path)

        if output_dir and output_dir != onnx_path.parent:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / onnx_path.name
            shutil.copy2(onnx_path, dest)
            return dest

        return onnx_path
