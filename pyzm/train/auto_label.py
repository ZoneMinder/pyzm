"""Auto-labeling using existing pyzm Detector.

Runs a pre-trained YOLO model on uploaded images to pre-label known classes,
saving users from having to manually annotate every object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pyzm.train.dataset import Annotation

if TYPE_CHECKING:
    import numpy as np
    from pyzm.models.detection import DetectionResult

logger = logging.getLogger("pyzm.train")


def _bbox_to_annotation(
    x1: int, y1: int, x2: int, y2: int,
    class_id: int,
    img_w: int, img_h: int,
) -> Annotation:
    """Convert pixel bbox to YOLO normalised annotation."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return Annotation(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def detections_to_annotations(
    result: "DetectionResult",
    target_classes: list[str],
    img_w: int,
    img_h: int,
) -> list[Annotation]:
    """Convert DetectionResult detections to Annotation list.

    Only detections whose label is in *target_classes* are kept.
    The class_id is the index in *target_classes*.
    """
    annotations: list[Annotation] = []
    class_map = {name.lower(): idx for idx, name in enumerate(target_classes)}
    for det in result.detections:
        label_lower = det.label.lower()
        if label_lower not in class_map:
            continue
        b = det.bbox
        ann = _bbox_to_annotation(
            b.x1, b.y1, b.x2, b.y2,
            class_id=class_map[label_lower],
            img_w=img_w, img_h=img_h,
        )
        annotations.append(ann)
    return annotations


def auto_label(
    image_paths: list[Path],
    base_path: str,
    processor: str,
    target_classes: list[str],
) -> dict[Path, list[Annotation]]:
    """Run existing YOLO model on images and return pre-annotations.

    Parameters
    ----------
    image_paths:
        Images to auto-label.
    base_path:
        Model base path for pyzm Detector.
    processor:
        ``"cpu"`` or ``"gpu"``.
    target_classes:
        Ordered class names.  Only detections matching these are returned.

    Returns
    -------
    dict mapping image path -> list of Annotations.
    """
    import cv2
    from pyzm.ml.detector import Detector

    det = Detector(base_path=base_path, processor=processor)

    results: dict[Path, list[Annotation]] = {}
    for img_path in image_paths:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image: %s", img_path)
            results[img_path] = []
            continue

        h, w = img.shape[:2]
        try:
            det_result = det.detect(img)
        except Exception:
            logger.exception("Detection failed for %s", img_path)
            results[img_path] = []
            continue

        results[img_path] = detections_to_annotations(
            det_result, target_classes, img_w=w, img_h=h,
        )
        logger.info(
            "Auto-labeled %s: %d annotations",
            img_path.name, len(results[img_path]),
        )

    return results
