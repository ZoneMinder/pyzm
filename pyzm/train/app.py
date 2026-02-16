"""Streamlit UI for YOLO fine-tuning (problem-driven workflow).

Launch with::

    python -m pyzm.train
    streamlit run pyzm/train/app.py -- --base-path /path/to/models

Phases (sidebar-driven):
    1. Select Images -- import frames from ZM events, YOLO datasets, or raw images
    2. Review Detections -- approve/edit/delete auto-detected objects
    3. Train & Export -- fine-tune and export ONNX
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# streamlit-drawable-canvas 0.9.x calls st.elements.image.image_to_url which
# was removed in Streamlit >=1.39.  Provide a shim.
import streamlit.elements.image as _st_image
if not hasattr(_st_image, "image_to_url"):
    def _image_to_url(
        image, width, clamp, channels, output_format, image_id, **_kw
    ):
        from streamlit.runtime import get_instance
        buf = BytesIO()
        fmt = output_format or "PNG"
        image.save(buf, format=fmt)
        data = buf.getvalue()
        mimetype = f"image/{fmt.lower()}"
        mgr = get_instance().media_file_mgr
        return mgr.add(data, mimetype, image_id)
    _st_image.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas

from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.trainer import ClassMetrics, HardwareInfo, TrainProgress, TrainResult, YOLOTrainer
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")

DEFAULT_BASE_PATH = "/var/lib/zmeventnotification/models"
PROJECTS_ROOT = Path.home() / ".pyzm" / "training"
MIN_IMAGES_PER_CLASS = 10

_COLOR_PALETTE = [
    "#27AE60", "#8E44AD", "#0081FE", "#FE3C71",
    "#F38630", "#5BB12F", "#E74C3C", "#3498DB",
]
_STATUS_COLORS = {
    DetectionStatus.PENDING: "#F1C40F",
    DetectionStatus.APPROVED: "#27AE60",
    DetectionStatus.DELETED: "#95A5A6",
    DetectionStatus.RENAMED: "#3498DB",
    DetectionStatus.RESHAPED: "#E67E22",
    DetectionStatus.ADDED: "#9B59B6",
}


# ===================================================================
# Compact CSS
# ===================================================================

def _inject_css() -> None:
    st.markdown("""<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    section[data-testid="stSidebar"] { min-width: 260px; max-width: 300px; }
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left; width: 100%; font-size: 0.85em; padding: 0.3rem 0.5rem;
    }
    h1, h2, h3 { font-size: 1.1em !important; margin-bottom: 0.3rem !important; }
    .stCaption { font-size: 0.78em !important; }
    .stCheckbox label, .stRadio label { font-size: 0.85em; }
    /* Hide Streamlit chrome but keep sidebar toggle */
    #MainMenu { display: none; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        height: auto !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] > * { pointer-events: auto; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { display: none; }
    /* Hide image/element toolbar (download, fullscreen, share, etc.) */
    [data-testid="stElementToolbar"] {
        display: none !important;
    }
    </style>""", unsafe_allow_html=True)


# ===================================================================
# Helpers
# ===================================================================

import re as _re

def _friendly_image_name(stem: str) -> str:
    """Turn filenames like ``event491_frame10`` into ``Event 491, Frame 10``."""
    m = _re.match(r"event(\d+)_frame(\d+)", stem)
    if m:
        return f"Event {m.group(1)}, Frame {m.group(2)}"
    return stem[:25]


# ===================================================================
# CLI args
# ===================================================================

def _parse_app_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    ap.add_argument("--workspace-dir", default=None)
    ap.add_argument("--processor", default="gpu")
    args, _ = ap.parse_known_args()
    return args


# ===================================================================
# Model scanning utilities
# ===================================================================

def _scan_models(base_path: str) -> list[dict]:
    bp = Path(base_path)
    if not bp.exists():
        return [{"name": "yolo11s", "path": "yolo11s.pt (auto-download)"}]
    models: list[dict] = []
    for d in sorted(bp.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in (".onnx", ".pt"):
                models.append({"name": f.stem, "path": str(f)})
    if not models:
        models.append({"name": "yolo11s", "path": "yolo11s.pt (auto-download)"})
    return models


def _read_model_classes(model_path: str) -> list[str]:
    p = Path(model_path)
    if not p.exists() or p.suffix != ".onnx":
        return []
    try:
        import ast
        import onnx
        model = onnx.load(str(p))
        meta = {prop.key: prop.value for prop in model.metadata_props}
        if "names" in meta:
            names_dict = ast.literal_eval(meta["names"])
            return [names_dict[i] for i in sorted(names_dict)]
    except Exception:
        pass
    return []



def _load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ===================================================================
# Drawing helpers
# ===================================================================

def _draw_verified_image(
    pil_img: Image.Image,
    detections: list[VerifiedDetection],
    *,
    skip_det_id: str | None = None,
) -> Image.Image:
    """Draw numbered, color-coded boxes on a PIL image.

    Parameters
    ----------
    skip_det_id
        If set, omit this detection (used during reshape mode so the
        editable rect isn't drawn twice).
    """
    from PIL import ImageDraw, ImageFont

    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    num = 0
    for det in detections:
        if det.status == DetectionStatus.DELETED:
            continue
        num += 1
        if det.detection_id == skip_det_id:
            continue
        ann = det.effective_annotation
        color = _STATUS_COLORS.get(det.status, "#999999")
        label_text = f"#{num} {det.effective_label}"

        x1 = int((ann.cx - ann.w / 2) * w)
        y1 = int((ann.cy - ann.h / 2) * h)
        x2 = int((ann.cx + ann.w / 2) * w)
        y2 = int((ann.cy + ann.h / 2) * h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        bbox = font.getbbox(label_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 4)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=color)
        draw.text((x1 + 3, label_y + 1), label_text, fill="#FFFFFF", font=font)

    return img


# ===================================================================
# Auto-detect
# ===================================================================

def _auto_detect_image(
    image_path: Path,
    args: argparse.Namespace,
) -> list[VerifiedDetection]:
    """Run auto-detect on a single image and return PENDING VerifiedDetections."""
    base_model = st.session_state.get("base_model", "yolo11s")
    model_classes = st.session_state.get("model_class_names", [])
    pdir = st.session_state.get("workspace_dir")
    best_pt = Path(pdir) / "runs" / "train" / "weights" / "best.pt" if pdir else None
    has_trained = best_pt is not None and best_pt.exists()

    detections: list[VerifiedDetection] = []
    if not (model_classes or has_trained):
        return detections

    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return detections
        h, w = img.shape[:2]
        from pyzm.ml.detector import Detector
        model_to_use = str(best_pt) if has_trained else base_model
        det = Detector(
            models=[model_to_use],
            base_path=args.base_path,
            processor=args.processor,
        )
        result = det.detect(img)
        for j, d in enumerate(result.detections):
            b = d.bbox
            cx = ((b.x1 + b.x2) / 2) / w
            cy = ((b.y1 + b.y2) / 2) / h
            bw = (b.x2 - b.x1) / w
            bh = (b.y2 - b.y1) / h
            ann = Annotation(class_id=0, cx=cx, cy=cy, w=bw, h=bh)
            detections.append(VerifiedDetection(
                detection_id=f"det_{j}",
                original=ann,
                status=DetectionStatus.PENDING,
                original_label=d.label,
            ))
    except Exception as exc:
        logger.warning("Auto-detect failed for %s: %s", image_path.name, exc)

    return detections


# ===================================================================
# Upload panel
# ===================================================================

def _upload_panel(
    ds: YOLODataset,
    store: VerificationStore,
    args: argparse.Namespace,
    *,
    target_classes: list[str] | None = None,
    label: str = "Upload images where detection failed or needs improvement",
) -> None:
    if target_classes:
        st.caption(f"Upload images containing: **{', '.join(target_classes)}**")
    upload_key = st.session_state.get("_upload_key", 0)
    uploaded = st.file_uploader(
        label,
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{upload_key}",
    )
    if not uploaded:
        return

    # Phase 1: save all images to disk
    import_bar = st.progress(0, text="Importing images...")
    destinations: list[Path] = []
    for i, f in enumerate(uploaded):
        tmp = Path(tempfile.mkdtemp()) / f.name
        tmp.write_bytes(f.read())
        destinations.append(ds.add_image(tmp, []))
        import_bar.progress((i + 1) / len(uploaded), text=f"Importing {i + 1}/{len(uploaded)}")
    import_bar.empty()

    # Phase 2: run auto-detection on each saved image
    detect_bar = st.progress(0, text="Running detection...")
    total_dets = 0
    for i, dest in enumerate(destinations):
        detections = _auto_detect_image(dest, args)
        total_dets += len(detections)
        store.set(ImageVerification(
            image_name=dest.name,
            detections=detections,
            fully_reviewed=False,
        ))
        detect_bar.progress((i + 1) / len(destinations), text=f"Detecting {i + 1}/{len(destinations)}")

    store.save()
    st.session_state["_upload_key"] = upload_key + 1
    detect_bar.progress(1.0, text=f"Added {len(destinations)} images, {total_dets} detections")
    st.toast(f"Added {len(destinations)} images with {total_dets} auto-detections")
    st.rerun()


# ===================================================================
# SIDEBAR
# ===================================================================

def _sidebar(ds: YOLODataset | None, store: VerificationStore | None) -> str:
    """Render sidebar. Returns the active phase key."""
    with st.sidebar:
        st.markdown(
            "<div style='font-size:0.9em; font-weight:700; margin-bottom:0.5rem;'>pyZM Training</div>",
            unsafe_allow_html=True,
        )

        # --- Phase navigation ---
        images = ds.staged_images() if ds else []
        has_images = len(images) >= 1
        all_reviewed = (
            has_images
            and store is not None
            and store.pending_count() == 0
            and store.reviewed_images_count() >= len(images)
        )
        train_done = st.session_state.get("_train_shared", {}).get("result") is not None

        current = st.session_state.get("active_phase", "select")

        phases = [
            ("select",  "1. Select Images",          has_images),
            ("review",  "2. Review Detections",      all_reviewed),
            ("train",   "3. Train & Export",         train_done),
        ]
        for key, label, done in phases:
            prefix = "-> " if key == current else "   "
            check = " \u2713" if done else ""
            btn_label = f"{prefix}{label}{check}"
            if st.button(btn_label, key=f"phase_{key}", width="stretch"):
                st.session_state["active_phase"] = key
                st.session_state.pop("_auto_label", None)
                st.rerun()

        # --- Image list (during review phase) ---
        if current == "review" and images and store:
            st.divider()
            _sidebar_image_list(store, images)

        # --- Class coverage ---
        if store:
            classes = store.build_class_list()
            if classes:
                st.divider()
                st.markdown(
                    "<div style='font-size:0.8em; font-weight:bold;'>Class Coverage</div>",
                    unsafe_allow_html=True,
                )
                counts = store.per_class_image_counts(classes)
                for cls in classes:
                    count = counts.get(cls, 0)
                    pct = min(1.0, count / MIN_IMAGES_PER_CLASS)
                    ready = " ok" if count >= MIN_IMAGES_PER_CLASS else ""
                    st.caption(f"{cls}: {count}/{MIN_IMAGES_PER_CLASS}{ready}")
                    st.progress(pct)

        # --- Project actions ---
        st.divider()
        project_name = st.session_state.get("project_name", "")
        if project_name:
            st.caption(f"Project: **{project_name}**")
            st.caption(f"`{PROJECTS_ROOT / project_name}`")
        if st.button("Switch Project", key="switch_project"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        if st.button("Reset Project", key="reset_workspace"):
            st.session_state["_confirm_reset"] = True
            st.rerun()
        if st.session_state.get("_confirm_reset"):
            st.warning("Delete all images, annotations, and settings?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, reset", type="primary", key="reset_confirm"):
                    _reset_project()
            with c2:
                if st.button("Cancel", key="reset_cancel"):
                    st.session_state.pop("_confirm_reset", None)
                    st.rerun()

    return current


def _sidebar_image_list(store: VerificationStore, images: list[Path]) -> None:
    """Compact image navigator in the sidebar."""
    reviewed = sum(
        1 for img in images
        if (iv := store.get(img.name)) and iv.fully_reviewed
    )
    st.markdown(
        f"<div style='font-size:0.8em; font-weight:bold;'>"
        f"Images ({reviewed}/{len(images)} reviewed)</div>",
        unsafe_allow_html=True,
    )
    selected_idx = st.session_state.get("selected_image_idx", 0)

    with st.container(height=350):
        for i, img in enumerate(images):
            iv = store.get(img.name)
            if iv and iv.fully_reviewed:
                icon = "\u2713"
            elif iv and iv.pending_count > 0:
                icon = f"[{iv.pending_count}]"
            else:
                icon = ""

            is_active = i == selected_idx
            label = f"{_friendly_image_name(img.stem)} {icon}"
            if st.button(
                label, key=f"img_nav_{i}",
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                st.session_state["selected_image_idx"] = i
                st.rerun()



# ===================================================================
# PHASE 1: Select Images
# ===================================================================

def _phase_select(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    st.markdown("### Select Images")

    # Show banner when classes need more training images
    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
    if needs:
        summary = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in needs
        )
        st.info(f"Classes needing more images: {summary}")

    source = st.radio(
        "Data source",
        ["Pre-Annotated YOLO Dataset", "Raw Images", "ZoneMinder Events"],
        horizontal=True,
        key="data_source",
    )

    if source == "Pre-Annotated YOLO Dataset":
        st.caption("Import a pre-annotated dataset in YOLO format.")
        from pyzm.train.local_import import local_dataset_panel
        local_dataset_panel(ds, store, args)
    elif source == "Raw Images":
        st.caption("Import unannotated images for manual annotation.")
        from pyzm.train.local_import import raw_images_panel
        raw_images_panel(ds, store, args, auto_detect_fn=_auto_detect_image)
    else:
        st.caption("Select events where detection was wrong or missing.")
        from pyzm.train.zm_browser import zm_event_browser_panel
        zm_event_browser_panel(ds, store, args)

    images = ds.staged_images()
    if images:
        st.divider()
        st.success(f"{len(images)} image{'s' if len(images) != 1 else ''} imported. Ready for review.")
        with st.expander(f"Imported images ({len(images)})", expanded=False):
            for img in images:
                iv = store.get(img.name)
                status = "\u2713" if iv and iv.fully_reviewed else "\u23f3"
                det_count = len(iv.detections) if iv else 0
                st.caption(f"{status} {img.name} ({det_count} annotations)")
        if st.button("Go to Review", type="primary"):
            st.session_state["active_phase"] = "review"
            st.session_state.pop("_auto_label", None)
            st.rerun()


# ===================================================================
# PHASE 2: Review Detections
# ===================================================================

def _phase_review(ds: YOLODataset, store: VerificationStore) -> None:
    images = ds.staged_images()
    if not images:
        st.info("No images yet. Go to **Select Images** to add some.")
        return

    st.markdown("### Review Detections")

    selected_idx = st.session_state.get("selected_image_idx", 0)
    if selected_idx >= len(images):
        selected_idx = 0
        st.session_state["selected_image_idx"] = 0

    img_path = images[selected_idx]
    pil_img = _load_image_pil(img_path)
    iv = store.get(img_path.name)
    if iv is None:
        iv = ImageVerification(image_name=img_path.name)
        store.set(iv)

    image_name = img_path.name
    canvas_counter = st.session_state.get(f"_canvas_counter_{image_name}", 0)

    # --- Navigation bar ---
    nav1, nav2, nav3 = st.columns([1, 4, 1])
    with nav1:
        if st.button("Prev", disabled=selected_idx == 0, width="stretch"):
            st.session_state["selected_image_idx"] = selected_idx - 1
            st.rerun()
    with nav2:
        status_text = "Reviewed" if iv.fully_reviewed else f"{iv.pending_count} pending"
        status_color = _STATUS_COLORS[DetectionStatus.APPROVED] if iv.fully_reviewed else _STATUS_COLORS[DetectionStatus.PENDING]
        st.markdown(
            f"<div style='text-align:center; font-size:0.9em;'>"
            f"<b>{img_path.name}</b> ({selected_idx + 1}/{len(images)}) "
            f"<span style='color:{status_color}'>{status_text}</span></div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("Next", disabled=selected_idx >= len(images) - 1, width="stretch"):
            st.session_state["selected_image_idx"] = selected_idx + 1
            st.rerun()

    # --- Compute canvas dimensions ---
    img_w, img_h = pil_img.size
    expanded = st.session_state.get("_canvas_expanded", False)
    max_w = 1200 if expanded else 700
    scale = min(1.0, max_w / img_w)
    canvas_w = int(img_w * scale)
    canvas_h = int(img_h * scale)

    # --- Check modal states ---
    reshape_det_id = st.session_state.get(f"_reshape_{image_name}")
    pending_rects: list[dict] = st.session_state.get(f"_pending_rects_{image_name}", [])
    changed = False

    if reshape_det_id:
        # ---- RESHAPE MODE ----
        changed |= _canvas_reshape(
            pil_img, iv, image_name, reshape_det_id,
            canvas_w, canvas_h, canvas_counter,
        )
    elif pending_rects:
        # ---- LABEL NEW BOX MODE ----
        changed |= _canvas_label_pending(
            pil_img, iv, store, image_name, pending_rects,
            canvas_w, canvas_h, canvas_counter,
        )
    else:
        # ---- NORMAL MODE: interactive canvas ----
        # Toolbar above canvas
        tb1, tb2 = st.columns(2)
        with tb1:
            expand_label = "Shrink canvas" if expanded else "Expand canvas"
            if st.button(expand_label, key="toggle_canvas_expand"):
                st.session_state["_canvas_expanded"] = not expanded
                st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
                st.rerun()
        with tb2:
            if st.button("Clear drawn box", key="undo_canvas_draw"):
                st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
                st.rerun()

        auto_label = st.session_state.get("_auto_label")
        if auto_label:
            st.info(f"Draw a rectangle to add **{auto_label}**.")
        elif not iv.detections:
            st.info("Draw boxes on the image to mark objects.")
        else:
            st.caption("Draw a rectangle to add another detection.")

        bg_img = pil_img.resize((canvas_w, canvas_h))
        bg_img = _draw_verified_image(bg_img, iv.detections)

        canvas_result = st_canvas(
            fill_color="#9B59B622",
            stroke_width=3,
            stroke_color="#9B59B6",
            background_image=bg_img,
            drawing_mode="rect",
            height=canvas_h,
            width=canvas_w,
            key=f"canvas_{image_name}_{canvas_counter}",
        )

        # Detect newly drawn rectangles
        if canvas_result and canvas_result.json_data:
            new_rects = [
                obj for obj in canvas_result.json_data.get("objects", [])
                if obj.get("type") == "rect"
            ]
            if new_rects:
                if auto_label:
                    # Auto-assign label without prompting
                    _save_pending_rects(
                        iv, new_rects, auto_label,
                        canvas_w, canvas_h, image_name, canvas_counter,
                    )
                    changed = True
                else:
                    st.session_state[f"_pending_rects_{image_name}"] = new_rects
                    st.rerun()

    # --- Detection list (always visible unless reshaping) ---
    if not reshape_det_id:
        if iv.detections and not pending_rects:
            changed |= _detection_list(iv, store, image_name)

    # --- Primary action button ---
    if not reshape_det_id and not pending_rects:
        st.divider()
        pending = [d for d in iv.detections if d.status == DetectionStatus.PENDING]

        # Check if this is the last unreviewed image
        unreviewed_remaining = sum(
            1 for img in images
            if not ((v := store.get(img.name)) and v.fully_reviewed)
        )
        is_last = unreviewed_remaining <= 1  # current image is the last (or only)

        # If classes need more images, loop back to select; otherwise go to train
        needs_more = bool(store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS))
        next_phase = "select" if needs_more else "train"
        next_phase_label = "import more images" if needs_more else "train"

        if pending:
            if is_last:
                btn_label = f"Approve all ({len(pending)}) & continue"
            else:
                btn_label = f"Approve all ({len(pending)}) & next"
        elif iv.fully_reviewed:
            btn_label = "Next image" if not is_last else f"Continue to {next_phase_label}"
        else:
            btn_label = "Next image" if not is_last else f"Continue to {next_phase_label}"

        if st.button(btn_label, type="primary", width="stretch"):
            for d in iv.detections:
                if d.status == DetectionStatus.PENDING:
                    d.status = DetectionStatus.APPROVED
            iv.fully_reviewed = True
            store.set(iv)
            store.save()
            if is_last:
                st.session_state["active_phase"] = next_phase
            else:
                _advance_to_next_unreviewed(store, images, selected_idx)
            st.rerun()

    if changed:
        store.set(iv)
        store.save()
        st.rerun()

    # --- Post-review guidance (shown when all images are reviewed) ---
    all_reviewed = (
        store.pending_count() == 0
        and store.reviewed_images_count() >= len(images)
    )
    if all_reviewed and not reshape_det_id and not pending_rects:
        needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
        if needs:
            st.divider()
            names = ", ".join(f"**{e['class_name']}**" for e in needs)
            st.info(
                f"Classes needing more training images: {names}. "
                f"Upload at least **{MIN_IMAGES_PER_CLASS}** images for each."
            )
            if st.button("Import More Images", type="primary",
                         key="review_go_select"):
                st.session_state["active_phase"] = "select"
                st.rerun()


# -------------------------------------------------------------------
# Canvas sub-modes
# -------------------------------------------------------------------

def _canvas_reshape(
    pil_img: Image.Image,
    iv: ImageVerification,
    image_name: str,
    det_id: str,
    canvas_w: int,
    canvas_h: int,
    canvas_counter: int,
) -> bool:
    """Reshape mode: canvas in ``transform`` with the selected detection
    as a draggable/resizable rectangle."""
    det = next((d for d in iv.detections if d.detection_id == det_id), None)
    if det is None:
        st.session_state.pop(f"_reshape_{image_name}", None)
        return False

    # Background: all detections EXCEPT the one being reshaped
    bg_img = pil_img.resize((canvas_w, canvas_h))
    bg_img = _draw_verified_image(bg_img, iv.detections, skip_det_id=det_id)

    # Initial drawing: the detection to reshape
    ann = det.effective_annotation
    x = int((ann.cx - ann.w / 2) * canvas_w)
    y = int((ann.cy - ann.h / 2) * canvas_h)
    w = int(ann.w * canvas_w)
    h = int(ann.h * canvas_h)

    initial = {
        "version": "4.4.0",
        "objects": [{
            "type": "rect",
            "left": x, "top": y, "width": w, "height": h,
            "fill": "rgba(155,89,182,0.12)",
            "stroke": "#E67E22", "strokeWidth": 3,
            "scaleX": 1, "scaleY": 1,
        }],
    }

    st.markdown(
        f"<div style='font-size:0.85em; margin-bottom:4px;'>"
        f"Reshaping <b>#{_visible_index(iv, det_id)} {det.effective_label}</b> "
        f"&mdash; drag or resize the box, then save.</div>",
        unsafe_allow_html=True,
    )

    canvas_result = st_canvas(
        fill_color="#9B59B622",
        stroke_width=3,
        stroke_color="#E67E22",
        background_image=bg_img,
        drawing_mode="transform",
        initial_drawing=initial,
        height=canvas_h,
        width=canvas_w,
        key=f"canvas_reshape_{image_name}_{canvas_counter}",
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save shape", type="primary", width="stretch"):
            # Read the updated rect from canvas
            objs = (canvas_result.json_data or {}).get("objects", [])
            rect = next((o for o in objs if o.get("type") == "rect"), None)
            if rect:
                left = rect["left"]
                top = rect["top"]
                rw = rect["width"] * rect.get("scaleX", 1)
                rh = rect["height"] * rect.get("scaleY", 1)
                cx = max(0.0, min(1.0, (left + rw / 2) / canvas_w))
                cy = max(0.0, min(1.0, (top + rh / 2) / canvas_h))
                nw = max(0.0, min(1.0, rw / canvas_w))
                nh = max(0.0, min(1.0, rh / canvas_h))
                det.adjusted = Annotation(class_id=0, cx=cx, cy=cy, w=nw, h=nh)
                det.status = DetectionStatus.RESHAPED
            st.session_state.pop(f"_reshape_{image_name}", None)
            st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
            return True
    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.session_state.pop(f"_reshape_{image_name}", None)
            st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
            st.rerun()

    return False


def _canvas_label_pending(
    pil_img: Image.Image,
    iv: ImageVerification,
    store: VerificationStore,
    image_name: str,
    pending_rects: list[dict],
    canvas_w: int,
    canvas_h: int,
    canvas_counter: int,
) -> bool:
    """Show the image with the newly drawn box and a label-picker dialog."""
    from PIL import ImageDraw as _IDraw

    bg_img = pil_img.resize((canvas_w, canvas_h))
    bg_img = _draw_verified_image(bg_img, iv.detections)
    preview = bg_img.copy()
    drw = _IDraw.Draw(preview)
    for rect in pending_rects:
        x1 = int(rect["left"])
        y1 = int(rect["top"])
        x2 = x1 + int(rect["width"] * rect.get("scaleX", 1))
        y2 = y1 + int(rect["height"] * rect.get("scaleY", 1))
        drw.rectangle([x1, y1, x2, y2], outline="#9B59B6", width=3)
    st.image(preview, width=canvas_w)

    # Label dialog — text input + selectbox side by side; text input wins
    changed = False

    model_classes = st.session_state.get("model_class_names", [])
    user_labels = store.build_class_list()
    all_labels = sorted(set(model_classes) | set(user_labels))

    with st.form(key=f"lbl_form_{image_name}_{canvas_counter}"):
        c_new, c_existing = st.columns(2)
        with c_new:
            typed_label = st.text_input("Type a label name", placeholder="e.g. dog, car...")
        with c_existing:
            picked_label = st.selectbox(
                "Select existing label",
                options=[""] + all_labels,
                format_func=lambda x: x or "—",
            ) if all_labels else ""
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Apply", type="primary", width="stretch")
        with c2:
            cancelled = st.form_submit_button("Cancel", width="stretch")

    if submitted:
        final = typed_label.strip() if typed_label and typed_label.strip() else (picked_label or "")
        if final:
            _save_pending_rects(iv, pending_rects, final,
                               canvas_w, canvas_h, image_name, canvas_counter)
            changed = True
    elif cancelled:
        st.session_state[f"_pending_rects_{image_name}"] = []
        st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
        st.rerun()

    return changed


def _save_pending_rects(
    iv: ImageVerification,
    rects: list[dict],
    label: str,
    canvas_w: int,
    canvas_h: int,
    image_name: str,
    canvas_counter: int,
) -> None:
    """Convert pending canvas rectangles into VerifiedDetections and clear state."""
    existing_count = len(iv.detections)
    for j, rect in enumerate(rects):
        left = rect["left"]
        top = rect["top"]
        w = rect["width"] * rect.get("scaleX", 1)
        h = rect["height"] * rect.get("scaleY", 1)
        cx = max(0.0, min(1.0, (left + w / 2) / canvas_w))
        cy = max(0.0, min(1.0, (top + h / 2) / canvas_h))
        nw = max(0.0, min(1.0, w / canvas_w))
        nh = max(0.0, min(1.0, h / canvas_h))
        ann = Annotation(class_id=0, cx=cx, cy=cy, w=nw, h=nh)
        iv.detections.append(VerifiedDetection(
            detection_id=f"det_{existing_count + j}",
            original=ann,
            status=DetectionStatus.ADDED,
            original_label=label,
        ))
    st.session_state[f"_pending_rects_{image_name}"] = []
    st.session_state[f"_canvas_counter_{image_name}"] = canvas_counter + 1
    st.toast(f"Added {len(rects)} detection{'s' if len(rects) != 1 else ''}: {label}")


def _visible_index(iv: ImageVerification, det_id: str) -> int:
    """Return the 1-based display number for a detection (skipping DELETED)."""
    num = 0
    for d in iv.detections:
        if d.status == DetectionStatus.DELETED:
            continue
        num += 1
        if d.detection_id == det_id:
            return num
    return 0


def _detection_list(
    iv: ImageVerification,
    store: VerificationStore,
    image_name: str,
) -> bool:
    """Numbered detection list with actions.  Numbers match the image overlay."""
    changed = False
    known_labels = store.build_class_list()
    num = 0

    for det in iv.detections:
        if det.status == DetectionStatus.DELETED:
            # Show deleted ones dimmed, with undo
            st.markdown(
                f"<div style='font-size:0.8em; color:#95A5A6; margin:2px 0;'>"
                f"<s>{det.effective_label}</s> deleted</div>",
                unsafe_allow_html=True,
            )
            if st.button("Undo", key=f"undo_{image_name}_{det.detection_id}"):
                det.status = DetectionStatus.PENDING
                changed = True
            continue

        num += 1
        sc = _STATUS_COLORS.get(det.status, "#999")

        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; margin:6px 0 2px;'>"
            f"<span style='display:inline-block; width:12px; height:12px; "
            f"background:{sc}; border-radius:2px;'></span>"
            f"<span style='font-size:0.85em;'><b>#{num} {det.effective_label}</b> "
            f"<span style='color:{sc}'>[{det.status.value}]</span></span></div>",
            unsafe_allow_html=True,
        )

        col_a, col_d, col_s, col_r = st.columns(4)
        with col_a:
            if det.status in (DetectionStatus.APPROVED, DetectionStatus.ADDED):
                st.markdown(
                    "<span style='color:#27AE60; font-size:0.8em;'>Approved</span>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button("Approve", key=f"approve_{image_name}_{det.detection_id}",
                             width="stretch"):
                    det.status = DetectionStatus.APPROVED
                    changed = True
        with col_d:
            if st.button("Delete", key=f"delete_{image_name}_{det.detection_id}",
                         width="stretch"):
                det.status = DetectionStatus.DELETED
                changed = True
        with col_s:
            if st.button("Reshape", key=f"reshape_{image_name}_{det.detection_id}",
                         width="stretch"):
                st.session_state[f"_reshape_{image_name}"] = det.detection_id
                st.rerun()
        with col_r:
            rename_key = f"_renaming_{image_name}_{det.detection_id}"
            if st.button("Rename", key=f"rename_btn_{image_name}_{det.detection_id}",
                         width="stretch"):
                st.session_state[rename_key] = True
                st.rerun()

        # Rename input row (shown only when Rename is clicked)
        if st.session_state.get(f"_renaming_{image_name}_{det.detection_id}"):
            other_labels = sorted(set(known_labels) - {det.effective_label})

            with st.form(key=f"rename_form_{image_name}_{det.detection_id}"):
                c_new, c_existing = st.columns(2)
                with c_new:
                    ren_typed = st.text_input("Type a label name", placeholder="e.g. dog, car...")
                with c_existing:
                    ren_picked = st.selectbox(
                        "Select existing label",
                        options=[""] + other_labels,
                        format_func=lambda x: x or "—",
                    ) if other_labels else ""
                rc1, rc2 = st.columns(2)
                with rc1:
                    submitted = st.form_submit_button("Save", type="primary", width="stretch")
                with rc2:
                    cancel_rename = st.form_submit_button("Cancel", width="stretch")

            if submitted:
                final = ren_typed.strip() if ren_typed and ren_typed.strip() else (ren_picked or "")
                if final:
                    det.new_label = final
                    det.status = DetectionStatus.RENAMED
                    st.session_state.pop(f"_renaming_{image_name}_{det.detection_id}", None)
                    changed = True
            elif cancel_rename:
                st.session_state.pop(f"_renaming_{image_name}_{det.detection_id}", None)
                st.rerun()

    return changed


def _advance_to_next_unreviewed(
    store: VerificationStore,
    images: list[Path],
    current_idx: int,
) -> None:
    """Set selected_image_idx to the next unreviewed image after current."""
    for offset in range(1, len(images)):
        idx = (current_idx + offset) % len(images)
        iv = store.get(images[idx].name)
        if iv is None or not iv.fully_reviewed:
            st.session_state["selected_image_idx"] = idx
            return
    st.session_state["selected_image_idx"] = min(current_idx, len(images) - 1)


# ===================================================================
# PHASE 3: Train & Export
# ===================================================================

def _phase_train(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    st.markdown("### Train")
    pdir = st.session_state.get("workspace_dir")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    classes = store.build_class_list()
    if not classes:
        st.info("No verified detections yet. Go to **Review Detections** first.")
        return

    # Readiness check — only corrected classes need the image threshold
    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
    if needs:
        names = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in needs
        )
        st.warning(f"Need more images for: {names}")
        if st.button("Import More Images", key="train_go_select"):
            st.session_state["active_phase"] = "select"
            st.rerun()

    images = ds.staged_images()
    if len(images) < 2:
        st.info("Need at least 2 images for training.")
        return

    hw = YOLOTrainer.detect_hardware()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=300, value=50, step=10)
    with col2:
        batch = st.number_input("Batch", min_value=1, max_value=128, value=hw.suggested_batch)
    with col3:
        imgsz = st.selectbox("Image size", [416, 640], index=1)
    with col4:
        st.caption(hw.display)

    # Shared mutable dict — background thread mutates contents,
    # main thread reads.  Lives in session_state so it survives reruns.
    shared: dict = st.session_state.get("_train_shared", {})

    if not shared.get("active", False):
        all_ready = not needs
        if st.button("Start Training", type="primary", disabled=not all_ready):
            class_name_to_id = {c: i for i, c in enumerate(classes)}
            ds.set_classes(classes)

            for img_path in images:
                anns = store.finalized_annotations(img_path.name, class_name_to_id)
                ds.update_annotations(img_path.name, anns)

            ds.split()
            yaml_path = ds.generate_yaml()
            trainer = YOLOTrainer(
                base_model=base_model,
                project_dir=Path(pdir),
                device=hw.device,
            )
            shared = {
                "active": True,
                "progress": TrainProgress(total_epochs=epochs),
                "result": None,
                "log": [],
            }
            st.session_state["_train_shared"] = shared
            st.session_state["trainer"] = trainer
            st.session_state["classes"] = classes

            def _run(_s: dict = shared) -> None:
                def _cb(p: TrainProgress) -> None:
                    _s["progress"] = p

                # Capture ultralytics logger output
                class _TrainLogHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        log = _s["log"]
                        log.append(self.format(record))
                        if len(log) > 200:
                            del log[:-200]

                handler = _TrainLogHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                ul_logger = logging.getLogger("ultralytics")
                ul_logger.addHandler(handler)
                ul_logger.setLevel(logging.INFO)
                try:
                    r = trainer.train(
                        dataset_yaml=yaml_path, epochs=epochs,
                        batch=batch, imgsz=imgsz, progress_callback=_cb,
                    )
                    _s["result"] = r
                except Exception as exc:
                    _s["progress"] = TrainProgress(finished=True, error=str(exc))
                finally:
                    ul_logger.removeHandler(handler)
                    _s["active"] = False

            threading.Thread(target=_run, daemon=True).start()
            st.rerun()
    else:
        p: TrainProgress = shared.get("progress") or TrainProgress()
        if p.total_epochs > 0:
            pct = max(0.0, min(1.0, p.epoch / p.total_epochs))
            st.progress(pct, text=p.message or f"Epoch {p.epoch}/{p.total_epochs}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")
        m2.metric("Box Loss", f"{p.box_loss:.4f}")
        m3.metric("Cls Loss", f"{p.cls_loss:.4f}")
        m4.metric("mAP50", f"{p.mAP50:.3f}")

        if st.button("Stop"):
            trainer = st.session_state.get("trainer")
            if trainer:
                trainer.request_stop()

        # Live training log
        train_log = shared.get("log", [])
        if train_log:
            st.code("\n".join(train_log[-30:]), language=None)

        if p.error:
            st.error(p.error)
        elif not p.finished:
            import time
            time.sleep(2)
            st.rerun()

    # Results + Export
    result: TrainResult | None = shared.get("result")
    if result:
        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("mAP50", f"{result.final_mAP50:.3f}")
        r2.metric("mAP50-95", f"{result.final_mAP50_95:.3f}")
        r3.metric("Size", f"{result.model_size_mb:.1f} MB")
        r4.metric("Time", f"{result.elapsed_seconds / 60:.1f} min")

        best_ep_label = (
            f"Epoch {result.best_epoch}/{result.total_epochs}"
            if result.best_epoch > 0
            else f"{result.total_epochs} epochs"
        )
        st.caption(f"Best model from: **{best_ep_label}**")

        if result.per_class:
            import pandas as pd

            rows = []
            for cls_name, cm in sorted(result.per_class.items()):
                rows.append({
                    "Class": cls_name,
                    "Precision": f"{cm.precision:.3f}",
                    "Recall": f"{cm.recall:.3f}",
                    "AP@50": f"{cm.ap50:.3f}",
                    "AP@50-95": f"{cm.ap50_95:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if result.best_model:
            st.code(str(result.best_model), language=None)

        pdir = st.session_state.get("workspace_dir")
        if pdir:
            st.caption(f"Dataset: `{pdir}`")
            _training_analysis(result, Path(pdir) / "runs" / "train")

        _phase_export(args)


def _training_analysis(result: TrainResult, train_dir: Path) -> None:
    """Show interpretive guidance and diagnostic images after training."""
    st.markdown("#### Training Analysis")

    # -- (a) Interpretive guidance --
    mAP = result.final_mAP50
    if mAP >= 0.8:
        st.success(f"**Excellent** — mAP50 {mAP:.2f} indicates strong detection quality.")
    elif mAP >= 0.6:
        st.info(f"**Good** — mAP50 {mAP:.2f}. Model performs well; more data may push it higher.")
    elif mAP >= 0.3:
        st.warning(f"**Moderate** — mAP50 {mAP:.2f}. Consider adding more diverse training images.")
    else:
        st.warning(f"**Poor** — mAP50 {mAP:.2f}. The model needs significantly more training data.")

    # Per-class weak spots
    weak = [name for name, cm in result.per_class.items() if cm.ap50 < 0.5]
    if weak:
        st.info(
            f"**Weak classes** (AP50 < 0.5): {', '.join(sorted(weak))}. "
            "Consider adding more training images for these."
        )

    # Overfitting hint
    if (
        result.best_epoch > 0
        and result.total_epochs > 0
        and result.best_epoch < result.total_epochs * 0.5
    ):
        st.info(
            f"Best model was at epoch {result.best_epoch}/{result.total_epochs} "
            "(early in training). The model may have overfit — "
            "try fewer epochs or more training data."
        )

    # -- (b) Training curves --
    results_png = train_dir / "results.png"
    if results_png.exists():
        st.markdown("##### Training Curves")
        with st.expander("How to read these curves"):
            st.markdown(
                "**Loss curves** (box_loss, cls_loss, dfl_loss): should decrease "
                "and flatten. If training loss keeps dropping but validation loss "
                "rises, the model is overfitting — stop earlier or add more data.\n\n"
                "**Precision & Recall**: both should rise and stabilize near 1.0. "
                "Low precision = too many false detections; low recall = missing "
                "real objects.\n\n"
                "**mAP50 / mAP50-95**: the main quality scores — higher is better. "
                "A plateau means more epochs won't help; more diverse data will."
            )
        st.image(str(results_png), width="stretch")

    # -- (c) Confusion matrix --
    cm_norm = train_dir / "confusion_matrix_normalized.png"
    cm_plain = train_dir / "confusion_matrix.png"
    cm_path = cm_norm if cm_norm.exists() else cm_plain if cm_plain.exists() else None
    if cm_path:
        st.markdown("##### Confusion Matrix")
        with st.expander("How to read the confusion matrix"):
            st.markdown(
                "Each row is a **true** class, each column is a **predicted** class. "
                "Bright diagonal = correct predictions. Off-diagonal cells show "
                "which classes get confused with each other.\n\n"
                "A **background** row/column means missed detections (false negatives) "
                "or phantom detections (false positives). If a class has a high "
                "background score, the model needs more examples of that class."
            )
        st.image(str(cm_path), width="stretch")

    # -- (d) Evaluation curves --
    f1_path = train_dir / "F1_curve.png"
    pr_path = train_dir / "PR_curve.png"
    if f1_path.exists() or pr_path.exists():
        with st.expander("Evaluation Curves"):
            st.markdown(
                "**F1 Curve**: shows the balance between precision and recall at "
                "each confidence threshold. The peak is the optimal threshold — "
                "a sharp, high peak (close to 1.0) is ideal.\n\n"
                "**PR Curve**: precision vs. recall trade-off. A curve that hugs "
                "the top-right corner is a strong model. Area under the curve "
                "(AUC) equals AP — higher is better."
            )
            c1, c2 = st.columns(2)
            if f1_path.exists():
                c1.image(str(f1_path), caption="F1 Curve", width="stretch")
            if pr_path.exists():
                c2.image(str(pr_path), caption="PR Curve", width="stretch")

    # -- (e) Validation samples --
    val_labels = train_dir / "val_batch0_labels.jpg"
    val_preds = train_dir / "val_batch0_pred.jpg"
    if val_labels.exists() or val_preds.exists():
        with st.expander("Validation Samples"):
            st.markdown(
                "**Ground Truth** shows the actual labels. **Predictions** shows "
                "what the model detected. Compare them — missed objects or wrong "
                "labels indicate classes that need more training data."
            )
            c1, c2 = st.columns(2)
            if val_labels.exists():
                c1.image(str(val_labels), caption="Ground Truth", width="stretch")
            if val_preds.exists():
                c2.image(str(val_preds), caption="Predictions", width="stretch")


def _phase_export(args: argparse.Namespace) -> None:
    st.markdown("#### Export")
    pdir = st.session_state.get("workspace_dir")
    classes = st.session_state.get("classes", [])
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    best_pt = Path(pdir) / "runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        return

    suggested_name = f"{base_model}_finetune.onnx"
    export_path = st.text_input(
        "Export ONNX to",
        value=str(Path(args.base_path) / "custom_finetune" / suggested_name),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export ONNX", type="primary"):
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            with st.spinner("Exporting..."):
                try:
                    onnx_path = trainer.export_onnx(output_path=Path(export_path))
                    st.success(f"Exported: `{onnx_path}`")
                    st.code(
                        f"models:\n"
                        f"  - name: {onnx_path.stem}\n"
                        f"    type: object\n"
                        f"    framework: opencv\n"
                        f"    weights: {onnx_path}\n"
                        f"    min_confidence: 0.3\n"
                        f"    pattern: \"({'|'.join(classes)})\"\n",
                        language="yaml",
                    )
                except Exception as exc:
                    st.error(str(exc))

    with col2:
        test_file = st.file_uploader("Test image", type=["jpg", "jpeg", "png"], key="test_img")
        if test_file:
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            pil_img = Image.open(test_file).convert("RGB")
            img_array = np.array(pil_img)
            try:
                dets = trainer.evaluate(img_array[..., ::-1])
                from PIL import ImageDraw
                draw_img = pil_img.copy()
                draw = ImageDraw.Draw(draw_img)
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                    draw.text((x1, max(0, y1 - 12)), f"{d['label']} {d['confidence']:.0%}", fill="lime")
                st.image(draw_img, width="stretch")
                if dets:
                    st.caption(", ".join(f"{d['label']} {d['confidence']:.0%}" for d in dets))
                else:
                    st.caption("No detections")
            except Exception as exc:
                st.error(str(exc))


# ===================================================================
# Legacy label seeding
# ===================================================================

def _seed_from_legacy_labels(ds: YOLODataset, store: VerificationStore) -> None:
    images = ds.staged_images()
    classes = ds.classes
    seeded = 0
    for img_path in images:
        if store.get(img_path.name) is not None:
            continue
        anns = ds.annotations_for(img_path.name)
        if not anns:
            store.set(ImageVerification(image_name=img_path.name))
            continue
        detections = []
        for j, ann in enumerate(anns):
            label = classes[ann.class_id] if ann.class_id < len(classes) else f"class_{ann.class_id}"
            detections.append(VerifiedDetection(
                detection_id=f"det_{j}",
                original=ann,
                status=DetectionStatus.APPROVED,
                original_label=label,
            ))
        store.set(ImageVerification(
            image_name=img_path.name,
            detections=detections,
            fully_reviewed=True,
        ))
        seeded += 1
    if seeded:
        store.save()
        logger.info("Seeded %d images from legacy labels into VerificationStore", seeded)


# ===================================================================
# Main
# ===================================================================

def _reset_project() -> None:
    """Wipe current project and return to project selector."""
    import shutil

    pdir = st.session_state.get("workspace_dir")
    if pdir and Path(pdir).exists():
        shutil.rmtree(pdir)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _list_projects() -> list[dict]:
    """Return metadata for each project under PROJECTS_ROOT."""
    if not PROJECTS_ROOT.is_dir():
        return []

    projects = []
    for d in sorted(PROJECTS_ROOT.iterdir()):
        meta_path = d / "project.json"
        if d.is_dir() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
            # Count images
            images_all = d / "images" / "all"
            image_count = (
                sum(1 for p in images_all.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
                if images_all.is_dir() else 0
            )
            projects.append({
                "name": d.name,
                "path": d,
                "base_model": meta.get("base_model", ""),
                "classes": meta.get("classes", []),
                "image_count": image_count,
            })
    return projects


def _delete_all_projects() -> None:
    """Remove all projects under PROJECTS_ROOT."""
    import shutil
    if PROJECTS_ROOT.is_dir():
        shutil.rmtree(PROJECTS_ROOT)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _project_selector() -> Path | None:
    """Show project selection screen. Returns project dir or None."""
    st.markdown("### Projects")

    projects = _list_projects()

    if projects:
        st.markdown("**Resume an existing project:**")
        for proj in projects:
            col_name, col_info, col_btn = st.columns([3, 3, 1])
            with col_name:
                st.markdown(f"**{proj['name']}**")
            with col_info:
                parts = []
                if proj["image_count"]:
                    parts.append(f"{proj['image_count']} images")
                if proj["base_model"]:
                    parts.append(f"model: {proj['base_model']}")
                if proj["classes"]:
                    parts.append(f"{len(proj['classes'])} classes")
                st.caption(", ".join(parts) if parts else "empty")
            with col_btn:
                if st.button("Open", key=f"open_{proj['name']}", width="stretch"):
                    st.session_state["project_name"] = proj["name"]
                    st.rerun()

        st.divider()

    st.markdown("**Create a new project:**")
    col_input, col_create = st.columns([3, 1])
    with col_input:
        new_name = st.text_input(
            "Project name",
            placeholder="e.g. license_plates",
            label_visibility="collapsed",
        )
    with col_create:
        create_clicked = st.button("Create", type="primary", width="stretch")

    if create_clicked:
        name = (new_name or "").strip()
        if not name:
            st.error("Enter a project name.")
            return None
        # Sanitise: allow alphanumeric, dashes, underscores
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        pdir = PROJECTS_ROOT / safe_name
        if pdir.exists():
            st.error(f"Project '{safe_name}' already exists.")
            return None
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
        st.session_state["project_name"] = safe_name
        st.rerun()

    # Delete all projects
    if projects:
        st.divider()
        # Inject red styling for the next button
        st.markdown(
            "<style>#delete-all-section button { "
            "background-color: #E74C3C !important; "
            "border-color: #C0392B !important; color: white !important; }"
            "</style>"
            "<div id='delete-all-section'>",
            unsafe_allow_html=True,
        )
        if st.button("Delete All Projects", key="delete_all_projects"):
            st.session_state["_confirm_delete_all"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.get("_confirm_delete_all"):
            st.error("This will permanently delete **all** projects and their data.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete all", type="primary", key="confirm_delete_all"):
                    _delete_all_projects()
            with c2:
                if st.button("Cancel", key="cancel_delete_all"):
                    st.session_state.pop("_confirm_delete_all", None)
                    st.rerun()

    return None


def _ensure_project(project_name: str) -> Path:
    """Ensure the project directory exists and return its path."""
    pdir = PROJECTS_ROOT / project_name
    if not (pdir / "project.json").exists():
        ds = YOLODataset(project_dir=pdir, classes=[])
        ds.init_project()
    return pdir


def _model_picker(args: argparse.Namespace, pdir: Path) -> None:
    """Show model selection UI. Saves choice and moves to browse phase."""
    st.markdown("### Select Base Model")
    st.caption("Choose the model to fine-tune.")

    available = _scan_models(args.base_path)
    model_names = [m["name"] for m in available]
    model_paths = {m["name"]: m["path"] for m in available}

    default_idx = 0
    saved_model = st.session_state.get("base_model")
    for i, name in enumerate(model_names):
        if saved_model and name == saved_model:
            default_idx = i
            break
        if name == "yolo11s":
            default_idx = i

    base_model = st.selectbox(
        "Base model",
        options=model_names,
        index=default_idx,
        format_func=lambda n: f"{n}  ({model_paths[n]})",
    )

    model_path = model_paths.get(base_model, "")
    model_classes = _read_model_classes(model_path)
    if model_classes:
        st.caption(f"This model detects **{len(model_classes)}** classes.")
    else:
        st.caption("Could not read classes from model metadata.")

    if st.button("Start", type="primary"):
        meta_path = pdir / "project.json"
        meta = json.loads(meta_path.read_text())
        meta["base_model"] = base_model
        meta_path.write_text(json.dumps(meta, indent=2))

        st.session_state["base_model"] = base_model
        st.session_state["model_class_names"] = model_classes
        st.session_state["active_phase"] = "select"
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="pyZM: Customize your own ML model", layout="wide")
    _inject_css()

    args = _parse_app_args()

    # --- Project selection ---
    # If --workspace-dir is passed, skip project selector (backward compat)
    if args.workspace_dir:
        pdir = Path(args.workspace_dir)
        if not (pdir / "project.json").exists():
            ds = YOLODataset(project_dir=pdir, classes=[])
            ds.init_project()
        st.session_state["project_name"] = pdir.name
    elif not st.session_state.get("project_name"):
        _project_selector()
        return
    else:
        pdir = _ensure_project(st.session_state["project_name"])

    st.session_state["workspace_dir"] = str(pdir)
    ds = YOLODataset.load(pdir)
    store = VerificationStore(pdir)

    # First run: pick a base model before proceeding
    if not st.session_state.get("base_model"):
        # Try to load from saved metadata
        meta_path = pdir / "project.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("base_model"):
                st.session_state["base_model"] = meta["base_model"]
        if not st.session_state.get("base_model"):
            _sidebar(None, None)
            _model_picker(args, pdir)
            return

    # Load model class names into session if not present
    if "model_class_names" not in st.session_state:
        base_model = st.session_state.get("base_model", "yolo11s")
        available = _scan_models(args.base_path)
        model_paths = {m["name"]: m["path"] for m in available}
        model_path = model_paths.get(base_model, "")
        st.session_state["model_class_names"] = _read_model_classes(model_path)

    _seed_from_legacy_labels(ds, store)

    # Sidebar controls everything
    phase = _sidebar(ds, store)

    # Map stale "upload" phase to "select" (phase was removed)
    if phase == "upload":
        phase = "select"
        st.session_state["active_phase"] = "select"

    # Render active phase
    if phase == "select":
        _phase_select(ds, store, args)
    elif phase == "review":
        _phase_review(ds, store)
    elif phase == "train":
        _phase_train(ds, store, args)


if __name__ == "__main__":
    main()
