"""YOLO-format dataset management for fine-tuning.

Handles image/label storage, train/val splitting, YAML generation,
and dataset quality checks.

Directory layout::

    <project_dir>/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    ├── dataset.yaml
    └── project.json
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pyzm.train")


@dataclass
class Annotation:
    """Single bounding-box annotation in YOLO normalised format."""

    class_id: int
    cx: float  # centre-x, normalised 0-1
    cy: float  # centre-y, normalised 0-1
    w: float   # width, normalised 0-1
    h: float   # height, normalised 0-1

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> Annotation:
        parts = line.strip().split()
        return cls(
            class_id=int(parts[0]),
            cx=float(parts[1]),
            cy=float(parts[2]),
            w=float(parts[3]),
            h=float(parts[4]),
        )


@dataclass
class QualityWarning:
    """A single dataset quality issue."""

    level: str   # "warning" or "error"
    message: str


@dataclass
class QualityReport:
    """Dataset quality assessment."""

    total_images: int = 0
    annotated_images: int = 0
    unannotated_images: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    warnings: list[QualityWarning] = field(default_factory=list)


class YOLODataset:
    """Manage a YOLO-format dataset within a project directory.

    Parameters
    ----------
    project_dir : Path
        Root of the training project (e.g. ``~/.pyzm/training/my_project``).
    classes : list[str]
        Ordered class names.  Index in this list == class_id in labels.
    """

    def __init__(
        self,
        project_dir: Path,
        classes: list[str],
        class_groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.classes = list(classes)
        self.class_groups: dict[str, list[str]] = class_groups or {}

        # Unsplit staging area
        self._images_dir = self.project_dir / "images" / "all"
        self._labels_dir = self.project_dir / "labels" / "all"

        # Train/val dirs
        self._train_images = self.project_dir / "images" / "train"
        self._train_labels = self.project_dir / "labels" / "train"
        self._val_images = self.project_dir / "images" / "val"
        self._val_labels = self.project_dir / "labels" / "val"

    # ------------------------------------------------------------------
    # Project initialisation
    # ------------------------------------------------------------------

    def init_project(self) -> None:
        """Create directory structure and save project metadata."""
        for d in (
            self._images_dir, self._labels_dir,
            self._train_images, self._train_labels,
            self._val_images, self._val_labels,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self._save_project_json()

    def _save_project_json(self) -> None:
        meta = {
            "classes": self.classes,
            "class_groups": self.class_groups,
        }
        (self.project_dir / "project.json").write_text(
            json.dumps(meta, indent=2)
        )

    def set_classes(
        self,
        classes: list[str],
        class_groups: dict[str, list[str]] | None = None,
    ) -> None:
        """Replace the class list (and optionally class groups) and persist."""
        self.classes = list(classes)
        if class_groups is not None:
            self.class_groups = class_groups
        self._save_project_json()

    @classmethod
    def load(cls, project_dir: Path) -> YOLODataset:
        """Load an existing project from its ``project.json``."""
        meta = json.loads((Path(project_dir) / "project.json").read_text())
        return cls(
            project_dir=project_dir,
            classes=meta["classes"],
            class_groups=meta.get("class_groups", {}),
        )

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------

    def add_image(
        self,
        image_path: Path,
        annotations: list[Annotation],
    ) -> Path:
        """Copy *image_path* into the staging area and write its label file.

        Returns the destination image path.
        """
        image_path = Path(image_path)
        dest = self._images_dir / image_path.name

        # Avoid overwriting — append a counter if name collides
        if dest.exists() and not dest.samefile(image_path):
            stem = image_path.stem
            suffix = image_path.suffix
            counter = 1
            while dest.exists():
                dest = self._images_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if not dest.exists():
            shutil.copy2(image_path, dest)

        label_path = self._labels_dir / f"{dest.stem}.txt"
        label_path.write_text(
            "\n".join(a.to_yolo_line() for a in annotations) + "\n"
            if annotations
            else ""
        )
        return dest

    def update_annotations(
        self,
        image_name: str,
        annotations: list[Annotation],
    ) -> None:
        """Overwrite the label file for an existing image in staging."""
        label_path = self._labels_dir / f"{Path(image_name).stem}.txt"
        label_path.write_text(
            "\n".join(a.to_yolo_line() for a in annotations) + "\n"
            if annotations
            else ""
        )

    # ------------------------------------------------------------------
    # Image listing
    # ------------------------------------------------------------------

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def staged_images(self) -> list[Path]:
        """Return all images in the staging (``all``) directory."""
        if not self._images_dir.exists():
            return []
        return sorted(
            p for p in self._images_dir.iterdir()
            if p.suffix.lower() in self._IMG_EXTS
        )

    def annotations_for(self, image_name: str) -> list[Annotation]:
        """Load annotations for a staged image by filename."""
        label_path = self._labels_dir / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            return []
        lines = label_path.read_text().strip().splitlines()
        return [Annotation.from_yolo_line(ln) for ln in lines if ln.strip()]

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------

    def split(self, val_ratio: float = 0.2, seed: int = 42) -> None:
        """Split staged images into train/val sets.

        Clears any previous split, then copies from ``all/`` into
        ``train/`` and ``val/``.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1 (exclusive)")

        # Clean previous split
        for d in (
            self._train_images, self._train_labels,
            self._val_images, self._val_labels,
        ):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        images = self.staged_images()
        if not images:
            logger.warning("No images to split")
            return

        rng = random.Random(seed)
        shuffled = list(images)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_set = set(shuffled[:n_val])

        for img in shuffled:
            label_src = self._labels_dir / f"{img.stem}.txt"
            if img in val_set:
                shutil.copy2(img, self._val_images / img.name)
                if label_src.exists():
                    shutil.copy2(label_src, self._val_labels / f"{img.stem}.txt")
            else:
                shutil.copy2(img, self._train_images / img.name)
                if label_src.exists():
                    shutil.copy2(label_src, self._train_labels / f"{img.stem}.txt")

        logger.info(
            "Split: %d train, %d val (ratio=%.2f)",
            len(shuffled) - n_val, n_val, val_ratio,
        )

    # ------------------------------------------------------------------
    # dataset.yaml generation
    # ------------------------------------------------------------------

    def generate_yaml(self) -> Path:
        """Write ``dataset.yaml`` for Ultralytics and return its path."""
        yaml_path = self.project_dir / "dataset.yaml"

        names_block = "\n".join(
            f"  {i}: {name}" for i, name in enumerate(self.classes)
        )
        content = (
            f"path: {self.project_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"\n"
            f"nc: {len(self.classes)}\n"
            f"names:\n{names_block}\n"
        )
        yaml_path.write_text(content)
        return yaml_path

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------

    def quality_report(self) -> QualityReport:
        """Analyse dataset and return quality warnings."""
        images = self.staged_images()
        per_class: dict[str, int] = {cls: 0 for cls in self.classes}
        annotated = 0
        unannotated = 0

        for img in images:
            anns = self.annotations_for(img.name)
            if anns:
                annotated += 1
                for a in anns:
                    if 0 <= a.class_id < len(self.classes):
                        per_class[self.classes[a.class_id]] += 1
            else:
                unannotated += 1

        report = QualityReport(
            total_images=len(images),
            annotated_images=annotated,
            unannotated_images=unannotated,
            per_class=per_class,
        )

        # Generate warnings
        if len(images) < 20:
            report.warnings.append(QualityWarning(
                level="warning",
                message="Very small dataset. Consider adding more images.",
            ))

        if unannotated > 0:
            report.warnings.append(QualityWarning(
                level="warning",
                message=f"{unannotated} images have no annotations — will be skipped.",
            ))

        counts = [c for c in per_class.values() if c > 0]
        if counts:
            max_count = max(per_class.values())
            for cls, count in per_class.items():
                if count == 0:
                    continue
                if count < 10:
                    report.warnings.append(QualityWarning(
                        level="warning",
                        message=(
                            f"Very few images for '{cls}'. "
                            "Model will likely not detect this reliably."
                        ),
                    ))
                elif count < 30:
                    report.warnings.append(QualityWarning(
                        level="warning",
                        message=(
                            f"Limited images for '{cls}'. "
                            "50+ recommended for good accuracy."
                        ),
                    ))

                # Imbalance check
                if count > 0 and max_count / count > 3:
                    dominant = max(per_class, key=per_class.get)  # type: ignore[arg-type]
                    report.warnings.append(QualityWarning(
                        level="warning",
                        message=(
                            f"Class imbalance: '{dominant}' has "
                            f"{max_count / count:.0f}x more images than '{cls}'."
                        ),
                    ))

        return report
