"""Face recognition (dlib) backend adapter wrapping ``pyzm.ml.face``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyzm.ml.backends.base import MLBackend
from pyzm.models.config import ModelConfig
from pyzm.models.detection import BBox, Detection

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("pyzm.ml")


class FaceDlibBackend(MLBackend):
    """Wraps the legacy :class:`pyzm.ml.face.Face` facade (which delegates to
    ``pyzm.ml.face_dlib.FaceDlib`` or ``pyzm.ml.face_tpu.FaceTpu``).
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model: object | None = None

    # -- MLBackend interface --------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name or "face_dlib"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from pyzm.ml.face import Face  # lazy import

        options = self._config_to_legacy_options()
        logger.debug("%s: loading face recognition model", self.name)
        self._model = Face(options=options)

    def detect(self, image: "np.ndarray") -> list[Detection]:
        if self._model is None:
            self.load()

        assert self._model is not None
        # Legacy returns (boxes, labels, confidences, model_tags)
        # boxes are [left, top, right, bottom] (x1, y1, x2, y2)
        boxes, labels, confidences, _model_tags = self._model.detect(image=image)

        detections: list[Detection] = []
        for box, label, conf in zip(boxes, labels, confidences):
            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    model_name=self.name,
                    detection_type="face",
                )
            )
        return detections

    # -- internal helpers -----------------------------------------------------

    def _config_to_legacy_options(self) -> dict:
        """Map :class:`ModelConfig` fields to the dict keys the legacy Face
        code expects."""
        fw = self._config.framework.value
        # Legacy face code looks at 'face_detection_framework' which is
        # 'dlib' or 'tpu', not the full enum value 'face_dlib'/'face_tpu'.
        detection_fw = fw.replace("face_", "") if fw.startswith("face_") else fw

        return {
            "name": self.name,
            "face_detection_framework": detection_fw,
            "known_images_path": self._config.known_faces_dir,
            "face_model": self._config.face_model,
            "face_train_model": self._config.face_train_model,
            "face_recog_dist_threshold": self._config.face_recog_dist_threshold,
            "num_jitters": self._config.face_num_jitters,
            "upsample_times": self._config.face_upsample_times,
            "disable_locks": "yes" if self._config.disable_locks else "no",
            "auto_lock": not self._config.disable_locks,
            f"{self._config.processor.value}_max_processes": self._config.max_processes,
            f"{self._config.processor.value}_max_lock_wait": self._config.max_lock_wait,
        }
