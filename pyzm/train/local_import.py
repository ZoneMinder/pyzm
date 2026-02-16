"""Local dataset import for the training UI.

Supports two modes:

1. **Pre-annotated YOLO dataset** — folder with data.yaml + images/ + labels/,
   imported with all annotations pre-approved.
2. **Raw images** — unannotated images from a local folder, auto-detected and
   imported with ``fully_reviewed=False`` so they go through the Review phase.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import streamlit as st
import yaml

from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_yolo_folder(folder: Path) -> dict | str:
    """Validate a local YOLO dataset folder.

    Returns the parsed data.yaml dict on success, or an error string.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return f"Not a directory: {folder}"

    yaml_path = folder / "data.yaml"
    if not yaml_path.exists():
        return f"Missing data.yaml in {folder}"

    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as exc:
        return f"Failed to parse data.yaml: {exc}"

    if not isinstance(data, dict):
        return "data.yaml is not a valid YAML mapping"

    if "names" not in data:
        return "data.yaml missing required 'names' key"

    names = data["names"]
    if isinstance(names, list):
        data["names"] = {i: n for i, n in enumerate(names)}
    elif not isinstance(names, dict):
        return "'names' in data.yaml must be a list or dict"

    # Standard layout: top-level images/ + labels/
    images_dir = folder / "images"
    labels_dir = folder / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        data["_splits"] = [(images_dir, labels_dir)]
        return data

    # Roboflow-style split dirs: train/images/ + train/labels/, etc.
    splits: list[tuple[Path, Path]] = []
    for name in ("train", "valid", "val", "test"):
        si = folder / name / "images"
        sl = folder / name / "labels"
        if si.is_dir() and sl.is_dir():
            splits.append((si, sl))
    if splits:
        data["_splits"] = splits
        return data

    return f"No images/labels directories found in {folder}"


def _find_images(images_dir: Path) -> list[Path]:
    """Recursively find all image files under a directory."""
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )


def _find_matching_label(image_path: Path, images_dir: Path, labels_dir: Path) -> Path | None:
    """Find the label .txt file that corresponds to an image.

    Mirrors the relative path from images/ to labels/, swapping extension.
    """
    rel = image_path.relative_to(images_dir)
    label_path = labels_dir / rel.with_suffix(".txt")
    return label_path if label_path.exists() else None


def _infer_split(img_path: Path, images_dir: Path) -> str | None:
    """Infer the original train/val assignment from the directory layout.

    Returns ``"train"``, ``"val"``, or ``None`` (unknown).

    Supports:
    - Roboflow: ``train/images/img.jpg``, ``valid/images/img.jpg``
    - Standard subdirs: ``images/train/img.jpg``, ``images/val/img.jpg``
    - Flat layout (``images/img.jpg``) → ``None``
    """
    # Roboflow: images_dir is e.g. <root>/train/images — parent is the split
    parent_name = images_dir.parent.name.lower()
    if parent_name in ("train",):
        return "train"
    if parent_name in ("valid", "val"):
        return "val"

    # Standard subdirs: images_dir is e.g. <root>/images,
    # image might be at images/train/img.jpg
    try:
        rel = img_path.relative_to(images_dir)
    except ValueError:
        return None
    if rel.parts and len(rel.parts) > 1:
        first = rel.parts[0].lower()
        if first in ("train",):
            return "train"
        if first in ("valid", "val"):
            return "val"

    return None


def _import_local_dataset(
    ds: YOLODataset,
    store: VerificationStore,
    splits: list[tuple[Path, Path]],
    names_map: dict[int, str],
) -> tuple[int, int]:
    """Import images and labels from one or more (images_dir, labels_dir) pairs.

    Annotations are written into the dataset at import time (no empty
    label files).  The original train/val split is preserved in
    ``split_map`` so that :meth:`YOLODataset.split` can honor it.

    Returns (image_count, detection_count).
    """
    # Collect all (image_path, images_dir, labels_dir) across splits
    all_files: list[tuple[Path, Path, Path]] = []
    for images_dir, labels_dir in splits:
        for img_path in _find_images(images_dir):
            all_files.append((img_path, images_dir, labels_dir))

    if not all_files:
        return 0, 0

    progress = st.progress(0, text="Importing local dataset...")
    img_count = 0
    det_count = 0
    split_map: dict[str, str] = {}

    for i, (img_path, images_dir, labels_dir) in enumerate(all_files):
        # Parse annotations BEFORE add_image so labels are populated on import
        annotations: list[Annotation] = []
        detections: list[VerifiedDetection] = []
        label_path = _find_matching_label(img_path, images_dir, labels_dir)
        if label_path:
            lines = label_path.read_text().strip().splitlines()
            for j, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = Annotation.from_yolo_line(line)
                except (ValueError, IndexError):
                    logger.warning("Skipping malformed line in %s: %s", label_path, line)
                    continue
                class_name = names_map.get(ann.class_id, f"class_{ann.class_id}")
                annotations.append(ann)
                detections.append(VerifiedDetection(
                    detection_id=f"det_{j}",
                    original=ann,
                    status=DetectionStatus.APPROVED,
                    original_label=class_name,
                ))
                det_count += 1

        dest = ds.add_image(img_path, annotations)
        img_count += 1

        # Record original split assignment
        orig_split = _infer_split(img_path, images_dir)
        if orig_split:
            split_map[dest.name] = orig_split

        iv = ImageVerification(
            image_name=dest.name,
            detections=detections,
            fully_reviewed=True,
        )
        store.set(iv)
        progress.progress(
            (i + 1) / len(all_files),
            text=f"Importing... {i + 1}/{len(all_files)}",
        )

    # Persist split map and import classes
    if split_map:
        ds.set_setting("split_map", split_map)
    ds.set_setting("import_classes", sorted(names_map.values()))

    store.save()
    progress.progress(1.0, text=f"Imported {img_count} images, {det_count} annotations")
    return img_count, det_count


def local_dataset_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args,
) -> None:
    """Streamlit panel for importing a local YOLO dataset."""
    folder_path = st.text_input(
        "Path to YOLO dataset folder",
        placeholder="/path/to/my_dataset",
        help=(
            "Folder must contain data.yaml plus images/ and labels/ dirs. "
            "Standard (images/ + labels/) and Roboflow-style split "
            "(train/images/ + train/labels/, etc.) layouts are supported."
        ),
    )

    if not folder_path or not folder_path.strip():
        st.info(
            "Enter the path to a folder in standard YOLO format:\n"
            "```\n"
            "my_dataset/\n"
            "  data.yaml\n"
            "  images/\n"
            "  labels/\n"
            "```"
        )
        return

    folder = Path(folder_path.strip())

    col_validate, col_import = st.columns(2)
    with col_validate:
        validate_clicked = st.button("Validate", key="local_validate")

    # Use session state to persist validation results across reruns
    if validate_clicked:
        result = validate_yolo_folder(folder)
        if isinstance(result, str):
            st.session_state["_local_validation"] = {"error": result}
        else:
            names_map = result["names"]
            splits = result["_splits"]
            image_count = 0
            label_count = 0
            for images_dir, labels_dir in splits:
                imgs = _find_images(images_dir)
                image_count += len(imgs)
                label_count += sum(
                    1 for img in imgs
                    if _find_matching_label(img, images_dir, labels_dir) is not None
                )
            # Serialise splits as list of string pairs for session state
            st.session_state["_local_validation"] = {
                "folder": str(folder),
                "names_map": names_map,
                "splits": [(str(i), str(l)) for i, l in splits],
                "image_count": image_count,
                "label_count": label_count,
            }

    validation = st.session_state.get("_local_validation")
    if not validation:
        return

    if "error" in validation:
        st.error(validation["error"])
        return

    names_map = validation["names_map"]
    class_names = [names_map[k] for k in sorted(names_map)]
    st.success(
        f"Valid YOLO dataset: **{validation['image_count']}** images, "
        f"**{validation['label_count']}** label files, "
        f"**{len(class_names)}** classes"
    )
    st.caption(f"Classes: {', '.join(class_names)}")

    with col_import:
        import_clicked = st.button("Import", type="primary", key="local_import")

    if import_clicked:
        splits = [(Path(i), Path(l)) for i, l in validation["splits"]]
        img_count, det_count = _import_local_dataset(
            ds, store, splits, names_map,
        )
        st.session_state.pop("_local_validation", None)
        st.toast(f"Imported {img_count} images with {det_count} annotations")
        st.rerun()


# ===================================================================
# Raw (unannotated) image import
# ===================================================================

def _import_raw_images(
    ds: YOLODataset,
    store: VerificationStore,
    folder: Path,
) -> int:
    """Import unannotated images from *folder*.

    Detection is deferred to the review phase.
    Returns the number of imported images.
    """
    all_images = _find_images(folder)
    if not all_images:
        return 0

    progress = st.progress(0, text="Importing raw images...")
    img_count = 0

    for i, img_path in enumerate(all_images):
        dest = ds.add_image(img_path, [])
        img_count += 1

        store.set(ImageVerification(
            image_name=dest.name,
            detections=[],
            fully_reviewed=False,
        ))
        progress.progress(
            (i + 1) / len(all_images),
            text=f"Importing... {i + 1}/{len(all_images)}",
        )

    store.save()
    progress.progress(1.0, text=f"Imported {img_count} images")
    return img_count


def raw_images_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
) -> None:
    """Streamlit panel for importing raw (unannotated) images."""
    method = st.radio(
        "Import method",
        ["Upload Images", "Server Folder"],
        horizontal=True,
        key="raw_import_method",
    )

    if method == "Upload Images":
        from pyzm.train.app import _upload_panel
        _upload_panel(
            ds, store, args,
            label="Upload images that we will use to train",
        )
    else:
        folder_path = st.text_input(
            "Path to image folder",
            placeholder="/path/to/my_images",
            help="Folder containing image files (.jpg, .jpeg, .png, .bmp, .webp).",
            key="raw_folder_path",
        )

        if not folder_path or not folder_path.strip():
            st.info(
                "Enter the path to a folder of images. "
                "No labels or data.yaml needed — the model will auto-detect objects "
                "and you can review/correct in the next phase."
            )
        else:
            folder = Path(folder_path.strip())
            scan_key = "_raw_scan_result"

            col_scan, col_import = st.columns(2)
            with col_scan:
                if st.button("Scan", key="raw_scan"):
                    if not folder.is_dir():
                        st.session_state[scan_key] = {"error": f"Not a directory: {folder}"}
                    else:
                        found = _find_images(folder)
                        st.session_state[scan_key] = {
                            "folder": str(folder),
                            "count": len(found),
                        }

            scan = st.session_state.get(scan_key)
            if scan:
                if "error" in scan:
                    st.error(scan["error"])
                elif scan["count"] == 0:
                    st.warning(f"No image files found in `{scan['folder']}`.")
                else:
                    st.success(f"Found **{scan['count']}** images in `{scan['folder']}`.")
                    with col_import:
                        if st.button("Import", type="primary", key="raw_import"):
                            img_count = _import_raw_images(
                                ds, store, Path(scan["folder"]),
                            )
                            st.session_state.pop(scan_key, None)
                            st.toast(f"Imported {img_count} images")
                            st.rerun()
