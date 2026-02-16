"""Streamlit UI for YOLO fine-tuning (problem-driven workflow).

Launch with::

    python -m pyzm.train
    streamlit run pyzm/train/app.py -- --base-path /path/to/models

Phases (sidebar-driven):
    1. Browse Events -- import frames from ZM events where detection failed
    2. Review Detections -- approve/edit/delete auto-detected objects
    3. Upload Training Data -- targeted uploads for classes the model struggles with
    4. Train & Export -- fine-tune and export ONNX
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
from pyzm.train.trainer import HardwareInfo, TrainProgress, TrainResult, YOLOTrainer
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)

logger = logging.getLogger("pyzm.train")

DEFAULT_BASE_PATH = "/var/lib/zmeventnotification/models"
DEFAULT_WORKSPACE = Path.home() / ".pyzm" / "training"
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
    /* Hide image toolbar icons (download, expand, etc.) — they render black on dark bg */
    [data-testid="StyledFullScreenButton"],
    [data-testid="StyledImageDownloadButton"],
    button[title="View fullscreen"],
    button[title="Download"] {
        display: none !important;
    }
    </style>""", unsafe_allow_html=True)


# ===================================================================
# Debug log capture
# ===================================================================

_log_buffer: list[str] = []


class _UILogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _log_buffer.append(self.format(record))
        if len(_log_buffer) > 500:
            del _log_buffer[:-500]


_ui_log_handler: _UILogHandler | None = None


def _setup_log_capture() -> None:
    global _ui_log_handler
    if _ui_log_handler is not None:
        return
    _ui_log_handler = _UILogHandler()
    _ui_log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    # Attach to the root pyzm logger so all child loggers are captured.
    pyzm_logger = logging.getLogger("pyzm")
    pyzm_logger.addHandler(_ui_log_handler)
    pyzm_logger.setLevel(logging.DEBUG)
    # Ensure propagation is on (so pyzm.train, pyzm.zm, etc. bubble up).
    pyzm_logger.propagate = True


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
) -> None:
    if target_classes:
        st.caption(f"Upload images containing: **{', '.join(target_classes)}**")
    upload_key = st.session_state.get("_upload_key", 0)
    uploaded = st.file_uploader(
        "Upload images where detection failed or needs improvement",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=f"uploader_{upload_key}",
    )
    if not uploaded:
        return

    progress_bar = st.progress(0, text="Processing uploads...")
    added_count = 0
    total_dets = 0

    for i, f in enumerate(uploaded):
        tmp = Path(tempfile.mkdtemp()) / f.name
        tmp.write_bytes(f.read())
        dest = ds.add_image(tmp, [])
        added_count += 1

        detections = _auto_detect_image(dest, args)
        total_dets += len(detections)

        iv = ImageVerification(
            image_name=dest.name,
            detections=detections,
            fully_reviewed=False,
        )
        store.set(iv)
        progress_bar.progress((i + 1) / len(uploaded), text=f"Processed {i + 1}/{len(uploaded)}")

    store.save()
    st.session_state["_upload_key"] = upload_key + 1
    progress_bar.progress(1.0, text=f"Added {added_count} images, {total_dets} detections")
    st.toast(f"Added {added_count} images with {total_dets} auto-detections")
    st.rerun()


# ===================================================================
# SIDEBAR
# ===================================================================

def _sidebar(args: argparse.Namespace, ds: YOLODataset | None, store: VerificationStore | None) -> str:
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
        # Upload is complete when no corrections exist or all corrected
        # classes have enough images.
        if store is not None:
            _needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)
            upload_complete = all_reviewed and (
                not _needs
                or all(e["current_images"] >= e["target_images"] for e in _needs)
            )
        else:
            upload_complete = False
        train_done = st.session_state.get("_train_shared", {}).get("result") is not None

        current = st.session_state.get("active_phase", "select")

        phases = [
            ("select",  "1. Browse Events",         has_images),
            ("review",  "2. Review Detections",      all_reviewed),
            ("upload",  "3. Upload Training Data",   upload_complete),
            ("train",   "4. Train & Export",         train_done),
        ]
        for key, label, done in phases:
            icon = "  " if not done else "  "
            prefix = "-> " if key == current else "   "
            check = " [done]" if done else ""
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

        # --- Reset ---
        st.divider()
        if st.button("Reset", key="reset_workspace"):
            st.session_state["_confirm_reset"] = True
            st.rerun()
        if st.session_state.get("_confirm_reset"):
            st.warning("Delete all images, annotations, and settings?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, reset", type="primary", key="reset_confirm"):
                    _reset_workspace(args)
            with c2:
                if st.button("Cancel", key="reset_cancel"):
                    st.session_state.pop("_confirm_reset", None)
                    st.rerun()

        # --- Debug logs ---
        st.divider()
        with st.expander("Debug Logs", expanded=False):
            if _log_buffer:
                st.code("\n".join(_log_buffer[-100:]), language=None)
            else:
                st.caption("No logs yet.")

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
                icon = "[done]"
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
    st.markdown("### Browse ZM Events")
    st.caption("Select events where detection was wrong or missing.")

    from pyzm.train.zm_browser import zm_event_browser_panel
    zm_event_browser_panel(ds, store, args)

    images = ds.staged_images()
    if images:
        st.divider()
        st.success(f"{len(images)} image{'s' if len(images) != 1 else ''} imported. Ready for review.")
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
    scale = min(1.0, 700 / img_w)
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

        auto_label = st.session_state.get("_auto_label")
        if auto_label:
            st.info(f"Draw a rectangle to add **{auto_label}**.")
        else:
            st.info("Draw a rectangle on the image to add a new detection.")

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
        if not iv.detections and not pending_rects:
            st.info("No detections. Draw boxes on the image above to mark objects.")
        elif iv.detections:
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

        if pending:
            if is_last:
                btn_label = f"Approve all ({len(pending)}) & continue"
            else:
                btn_label = f"Approve all ({len(pending)}) & next"
        elif iv.fully_reviewed:
            btn_label = "Next image" if not is_last else "Continue to upload"
        else:
            btn_label = "Done, next image" if not is_last else "Done, continue to upload"

        if st.button(btn_label, type="primary", width="stretch"):
            for d in iv.detections:
                if d.status == DetectionStatus.PENDING:
                    d.status = DetectionStatus.APPROVED
            iv.fully_reviewed = True
            store.set(iv)
            store.save()
            if is_last:
                st.session_state["active_phase"] = "upload"
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
        st.divider()
        corrections = store.corrected_classes()
        # Filter out old wrong labels (only renamed_from / deleted)
        _NEGATIVE_ONLY = {"renamed_from", "deleted"}
        corrections = {
            cls: reasons for cls, reasons in corrections.items()
            if not set(reasons.keys()) <= _NEGATIVE_ONLY
        }
        if corrections:
            new_classes = []
            improved_classes = []
            for cls, reasons in sorted(corrections.items()):
                if set(reasons.keys()) == {"added"}:
                    new_classes.append(cls)
                else:
                    improved_classes.append(cls)

            parts = []
            if new_classes:
                parts.append(
                    f"New classes you defined: **{', '.join(new_classes)}**."
                )
            if improved_classes:
                parts.append(
                    f"The model needs improvement on: "
                    f"**{', '.join(improved_classes)}**."
                )
            parts.append(
                f"Next: upload at least **{MIN_IMAGES_PER_CLASS}** training "
                f"images for each class so the model can learn."
            )
            st.info(" ".join(parts))
            if st.button("Go to Upload Training Data", type="primary",
                         key="review_go_upload"):
                st.session_state["active_phase"] = "upload"
                st.rerun()
        else:
            st.success("All detections look correct! Proceed to training.")
            if st.button("Go to Train", type="primary", key="review_go_train"):
                st.session_state["active_phase"] = "train"
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
    st.image(preview, width="stretch")

    # Label dialog
    st.markdown(
        "<div style='border:1px solid #9B59B6; border-radius:6px; "
        "padding:8px 12px; margin:-4px 0 8px;'>"
        "<span style='font-size:0.85em; font-weight:bold;'>"
        "Label this object:</span></div>",
        unsafe_allow_html=True,
    )

    changed = False

    # Merge model classes + any user-defined labels into one list
    model_classes = st.session_state.get("model_class_names", [])
    user_labels = store.build_class_list()
    all_labels = sorted(set(model_classes) | set(user_labels))

    # Selectbox — applies immediately on selection (no form)
    if all_labels:
        pick = st.selectbox(
            "Choose existing label",
            options=[""] + all_labels,
            format_func=lambda x: "Select a label..." if x == "" else x,
            key=f"lbl_pick_{image_name}_{canvas_counter}",
        )
        if pick:
            _save_pending_rects(iv, pending_rects, pick,
                               canvas_w, canvas_h, image_name, canvas_counter)
            changed = True

    # Custom label — separate form so typing works reliably
    if not changed:
        with st.form(key=f"lbl_custom_form_{image_name}_{canvas_counter}"):
            custom_label = st.text_input(
                "Or type a new label", placeholder="Type new label name...",
            )
            submitted = st.form_submit_button("Apply", type="primary", width="stretch")
        if submitted and custom_label and custom_label.strip():
            _save_pending_rects(iv, pending_rects, custom_label.strip(),
                               canvas_w, canvas_h, image_name, canvas_counter)
            changed = True

    if not changed:
        if st.button("Cancel", key=f"cancel_draw_{image_name}_{canvas_counter}"):
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
            if other_labels:
                rcols = st.columns(min(len(other_labels), 4))
                for j, lbl in enumerate(other_labels):
                    with rcols[j % len(rcols)]:
                        if st.button(lbl, key=f"ren_pick_{image_name}_{det.detection_id}_{j}",
                                     width="stretch"):
                            det.new_label = lbl
                            det.status = DetectionStatus.RENAMED
                            st.session_state.pop(f"_renaming_{image_name}_{det.detection_id}", None)
                            changed = True
            with st.form(key=f"rename_form_{image_name}_{det.detection_id}"):
                rc1, rc2 = st.columns([3, 1])
                with rc1:
                    custom_label = st.text_input(
                        "New label",
                        label_visibility="collapsed",
                        placeholder="Type new label...",
                    )
                with rc2:
                    submitted = st.form_submit_button("Save", width="stretch")
                if submitted and custom_label and custom_label.strip():
                    det.new_label = custom_label.strip()
                    det.status = DetectionStatus.RENAMED
                    st.session_state.pop(f"_renaming_{image_name}_{det.detection_id}", None)
                    changed = True

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
# PHASE 3: Upload Training Data
# ===================================================================

def _format_correction_reasons(corrections: dict[str, int]) -> str:
    """Human-readable summary of why a class needs more training data."""
    parts = []
    if "renamed_to" in corrections:
        parts.append(f"renamed to this in {corrections['renamed_to']}")
    if "renamed_from" in corrections:
        parts.append(f"renamed from this in {corrections['renamed_from']}")
    if "reshaped" in corrections:
        parts.append(f"reshaped in {corrections['reshaped']}")
    if "deleted" in corrections:
        parts.append(f"deleted in {corrections['deleted']}")
    if "added" in corrections:
        parts.append(f"manually added in {corrections['added']}")
    return ", ".join(parts)


def _phase_upload(ds: YOLODataset, store: VerificationStore, args: argparse.Namespace) -> None:
    st.markdown("### Upload Training Data")

    needs = store.classes_needing_upload(min_images=MIN_IMAGES_PER_CLASS)

    if not needs:
        st.success("All classes have enough training data. Ready to train!")
        if st.button("Go to Train", type="primary", key="upload_go_train_done"):
            st.session_state["active_phase"] = "train"
            st.rerun()
        return

    # --- Per-class wizard ---
    wizard_idx = st.session_state.get("_upload_wizard_idx", 0)
    if wizard_idx >= len(needs):
        wizard_idx = len(needs) - 1
        st.session_state["_upload_wizard_idx"] = wizard_idx

    entry = needs[wizard_idx]
    cls = entry["class_name"]
    current = entry["current_images"]
    target = entry["target_images"]
    ready = current >= target
    corrections = entry["corrections"]

    # Step header
    st.markdown(f"**Class {wizard_idx + 1} of {len(needs)}: `{cls}`**")

    # Explanation based on correction type
    if "added" in corrections and len(corrections) == 1:
        st.info(
            f"You added **{cls}** as a new object class. "
            f"Upload at least **{target}** images that clearly show "
            f"**{cls}** objects.\n\n"
            f"After uploading, go to **Review** to draw bounding boxes "
            f"on each image."
        )
    else:
        reason_text = _format_correction_reasons(corrections)
        st.info(
            f"You corrected **{cls}** detections ({reason_text}). "
            f"Upload at least **{target}** images containing **{cls}** "
            f"objects.\n\n"
            f"The model will auto-detect on uploaded images — verify in "
            f"**Review**."
        )

    # Progress bar
    col_bar, col_count = st.columns([4, 1])
    with col_bar:
        pct = min(1.0, current / target) if target > 0 else 0.0
        st.progress(pct)
    with col_count:
        st.caption(f"{current}/{target}" + (" ok" if ready else ""))

    # Upload widget (targeted at this class)
    _upload_panel(ds, store, args, target_classes=[cls])

    # Guide to Review if uploaded images need annotation
    images = ds.staged_images()
    unreviewed = len(images) - store.reviewed_images_count()
    if unreviewed > 0 and current < target:
        st.warning(
            f"**{unreviewed}** uploaded image{'s' if unreviewed != 1 else ''} "
            f"need review. Go to **Review** to draw bounding boxes for "
            f"**{cls}**."
        )
        if st.button("Go to Review", key=f"upload_review_{wizard_idx}"):
            st.session_state["active_phase"] = "review"
            st.session_state["_auto_label"] = cls
            st.rerun()

    # --- Navigation between classes ---
    st.divider()
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if wizard_idx > 0:
            prev_cls = needs[wizard_idx - 1]["class_name"]
            if st.button(f"< {prev_cls}", key="upload_prev", width="stretch"):
                st.session_state["_upload_wizard_idx"] = wizard_idx - 1
                st.rerun()
    with nav3:
        if wizard_idx < len(needs) - 1:
            next_cls = needs[wizard_idx + 1]["class_name"]
            if st.button(f"{next_cls} >", key="upload_next", width="stretch"):
                st.session_state["_upload_wizard_idx"] = wizard_idx + 1
                st.rerun()

    # --- Overall status ---
    remaining = [e for e in needs if e["current_images"] < e["target_images"]]
    if remaining:
        names = ", ".join(
            f"**{e['class_name']}** ({e['current_images']}/{e['target_images']})"
            for e in remaining
        )
        st.caption(f"Still need: {names}")


# ===================================================================
# PHASE 4: Train & Export
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
        if st.button("Go to Upload Training Data", key="train_go_upload"):
            st.session_state["active_phase"] = "upload"
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
        if result.best_model:
            st.code(str(result.best_model), language=None)

        _phase_export(args)


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

def _reset_workspace(args: argparse.Namespace) -> None:
    """Wipe workspace and clear all session state."""
    import shutil

    pdir = Path(args.workspace_dir) if args.workspace_dir else DEFAULT_WORKSPACE
    if pdir.exists():
        shutil.rmtree(pdir)
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _ensure_workspace(args: argparse.Namespace) -> Path:
    """Auto-create and return the single training workspace directory."""
    pdir = Path(args.workspace_dir) if args.workspace_dir else DEFAULT_WORKSPACE
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
    _setup_log_capture()

    args = _parse_app_args()
    pdir = _ensure_workspace(args)
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
            _sidebar(args, None, None)
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
    phase = _sidebar(args, ds, store)

    # Render active phase
    if phase == "select":
        _phase_select(ds, store, args)
    elif phase == "review":
        _phase_review(ds, store)
    elif phase == "upload":
        _phase_upload(ds, store, args)
    elif phase == "train":
        _phase_train(ds, store, args)


if __name__ == "__main__":
    main()
