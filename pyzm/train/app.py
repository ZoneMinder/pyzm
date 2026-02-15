"""Streamlit UI for YOLO fine-tuning.

Launch with::

    python -m pyzm.train          # uses streamlit run internally
    streamlit run pyzm/train/app.py -- --base-path /path/to/models
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from pyzm.train.dataset import Annotation, YOLODataset
from pyzm.train.trainer import HardwareInfo, TrainProgress, TrainResult, YOLOTrainer

logger = logging.getLogger("pyzm.train")

DEFAULT_BASE_PATH = "/var/lib/zmeventnotification/models"
DEFAULT_PROJECT_ROOT = Path.home() / ".pyzm" / "training"

# ---------------------------------------------------------------------------
# Argument parsing (passed after `--` by __main__.py)
# ---------------------------------------------------------------------------


def _parse_app_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--processor", default="gpu")
    # Streamlit passes extra args; ignore them
    args, _ = ap.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLOR_PALETTE = [
    "#27AE60", "#8E44AD", "#0081FE", "#FE3C71",
    "#F38630", "#5BB12F", "#E74C3C", "#3498DB",
]


def _scan_models(base_path: str) -> list[str]:
    """Find available model names by scanning base_path for .onnx/.pt files."""
    bp = Path(base_path)
    if not bp.exists():
        return ["yolo11s"]
    names: list[str] = []
    for d in sorted(bp.iterdir()):
        if d.is_dir():
            has_model = any(
                f.suffix in (".onnx", ".pt")
                for f in d.iterdir() if f.is_file()
            )
            if has_model:
                names.append(d.name)
    return names or ["yolo11s"]


def _project_dir(project_root: Path, name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return project_root / safe


def _load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _annotations_to_canvas_rects(
    annotations: list[Annotation],
    classes: list[str],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Convert YOLO annotations to fabric.js rect objects for the canvas."""
    rects = []
    for ann in annotations:
        x = (ann.cx - ann.w / 2) * img_w
        y = (ann.cy - ann.h / 2) * img_h
        w = ann.w * img_w
        h = ann.h * img_h
        cls_name = classes[ann.class_id] if ann.class_id < len(classes) else "?"
        color = _COLOR_PALETTE[ann.class_id % len(_COLOR_PALETTE)]
        rects.append({
            "type": "rect",
            "left": x,
            "top": y,
            "width": w,
            "height": h,
            "fill": f"{color}33",
            "stroke": color,
            "strokeWidth": 2,
            "label": cls_name,
            "class_id": ann.class_id,
        })
    return rects


def _canvas_rects_to_annotations(
    canvas_result: dict,
    class_name_to_id: dict[str, int],
    selected_class: str,
    img_w: int,
    img_h: int,
) -> list[Annotation]:
    """Convert fabric.js canvas objects back to YOLO annotations."""
    annotations: list[Annotation] = []
    objects = canvas_result.get("objects", []) if canvas_result else []
    for obj in objects:
        if obj.get("type") != "rect":
            continue
        # Get class from stored metadata or fall back to selected class
        cls_name = obj.get("label", selected_class)
        cls_id = class_name_to_id.get(cls_name, class_name_to_id.get(selected_class, 0))

        left = obj["left"]
        top = obj["top"]
        w = obj["width"] * obj.get("scaleX", 1)
        h = obj["height"] * obj.get("scaleY", 1)

        cx = (left + w / 2) / img_w
        cy = (top + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        annotations.append(Annotation(class_id=cls_id, cx=cx, cy=cy, w=nw, h=nh))
    return annotations


# ===================================================================
# PAGE 1: Setup
# ===================================================================

def _page_setup(args: argparse.Namespace, project_root: Path) -> None:
    st.header("Project Setup")

    st.info(
        "**Getting started:**\n"
        "1. Name your project\n"
        "2. Pick a base YOLO model (the starting point for fine-tuning)\n"
        "3. List all classes you want to detect (include COCO classes to keep, "
        "like *person* and *car*, plus your new custom classes)\n"
        "4. Click **Create Project**\n\n"
        "**Image tips:** 20 images minimum per class, 50+ recommended. "
        "Use varied angles, lighting, and backgrounds for best results."
    )

    project_name = st.text_input(
        "Project name",
        value=st.session_state.get("project_name", ""),
        placeholder="e.g. my-packages",
    )

    available_models = _scan_models(args.base_path)
    base_model = st.selectbox(
        "Base model",
        options=available_models,
        index=0,
        help="Select a model to fine-tune. Models are scanned from your base_path. "
             "Plain names like 'yolo11s' will auto-download from Ultralytics.",
    )

    classes_input = st.text_input(
        "Class names (comma-separated)",
        value=st.session_state.get("classes_input", ""),
        placeholder="e.g. person, car, package",
        help="Include COCO classes you want to keep AND your new custom classes.",
    )

    if st.button("Create Project", type="primary"):
        if not project_name.strip():
            st.error("Please enter a project name.")
            return
        if not classes_input.strip():
            st.error("Please enter at least one class name.")
            return

        classes = [c.strip() for c in classes_input.split(",") if c.strip()]
        if len(classes) < 1:
            st.error("Please enter at least one class name.")
            return

        pdir = _project_dir(project_root, project_name)
        ds = YOLODataset(project_dir=pdir, classes=classes)
        ds.init_project()

        st.session_state["project_name"] = project_name
        st.session_state["project_dir"] = str(pdir)
        st.session_state["classes"] = classes
        st.session_state["classes_input"] = classes_input
        st.session_state["base_model"] = base_model

        st.success(f"Project created at `{pdir}`")

    # Show existing projects
    if project_root.exists():
        existing = [
            d.name for d in sorted(project_root.iterdir())
            if d.is_dir() and (d / "project.json").exists()
        ]
        if existing:
            st.divider()
            st.subheader("Existing Projects")
            selected = st.selectbox("Load project", ["(none)"] + existing)
            if selected != "(none)" and st.button("Load"):
                pdir = project_root / selected
                ds = YOLODataset.load(pdir)
                st.session_state["project_name"] = selected
                st.session_state["project_dir"] = str(pdir)
                st.session_state["classes"] = ds.classes
                st.session_state["classes_input"] = ", ".join(ds.classes)
                st.session_state["base_model"] = "yolo11s"
                st.rerun()


# ===================================================================
# PAGE 2: Upload & Label
# ===================================================================

def _page_upload_label(args: argparse.Namespace) -> None:
    st.header("Upload & Label")

    pdir = st.session_state.get("project_dir")
    classes = st.session_state.get("classes")
    if not pdir or not classes:
        st.warning("Please create or load a project first (Setup page).")
        return

    ds = YOLODataset(project_dir=Path(pdir), classes=classes)

    # --- Upload section ---
    uploaded = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            tmp = Path(tempfile.mkdtemp()) / f.name
            tmp.write_bytes(f.read())
            ds.add_image(tmp, [])
        st.success(f"Added {len(uploaded)} images.")
        st.rerun()

    # --- Auto-label ---
    images = ds.staged_images()
    if images and st.button("Auto-Label All Images"):
        with st.spinner("Running detection model on all images..."):
            try:
                from pyzm.train.auto_label import auto_label

                results = auto_label(
                    image_paths=images,
                    base_path=args.base_path,
                    processor=args.processor,
                    target_classes=classes,
                )
                for img_path, anns in results.items():
                    if anns:
                        ds.update_annotations(img_path.name, anns)
                st.success("Auto-labeling complete! Review and adjust boxes below.")
                st.rerun()
            except Exception as exc:
                st.error(f"Auto-labeling failed: {exc}")

    if not images:
        st.info("Upload some images to get started.")
        return

    # --- Quality panel (sidebar) ---
    with st.sidebar:
        st.subheader("Dataset Quality")
        report = ds.quality_report()
        st.metric("Total images", report.total_images)
        st.metric("Annotated", report.annotated_images)
        st.metric("Unannotated", report.unannotated_images)
        for cls, count in report.per_class.items():
            st.metric(f"Class: {cls}", count)
        for w in report.warnings:
            st.warning(w.message)

    # --- Image gallery ---
    st.subheader("Images")
    cols_per_row = 5
    selected_idx = st.session_state.get("selected_image_idx", 0)

    # Thumbnail gallery
    for row_start in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(images):
                break
            img_path = images[idx]
            anns = ds.annotations_for(img_path.name)
            with col:
                thumb = _load_image_pil(img_path)
                thumb.thumbnail((120, 120))
                st.image(thumb, use_container_width=True)
                badge = f"**{len(anns)} labels**" if anns else "*unlabeled*"
                if st.button(badge, key=f"sel_{idx}"):
                    st.session_state["selected_image_idx"] = idx
                    st.rerun()

    # --- Annotation canvas ---
    if selected_idx < len(images):
        st.divider()
        st.subheader("Annotate")

        img_path = images[selected_idx]
        pil_img = _load_image_pil(img_path)
        img_w, img_h = pil_img.size

        # Scale for canvas (max 700px wide)
        scale = min(1.0, 700 / img_w)
        canvas_w = int(img_w * scale)
        canvas_h = int(img_h * scale)

        class_name_to_id = {c: i for i, c in enumerate(classes)}
        selected_class = st.selectbox(
            "Class for new boxes",
            options=classes,
            key="annotation_class",
        )
        color_idx = class_name_to_id[selected_class]
        stroke_color = _COLOR_PALETTE[color_idx % len(_COLOR_PALETTE)]

        # Prepare initial rects from existing annotations
        existing_anns = ds.annotations_for(img_path.name)
        initial_rects = _annotations_to_canvas_rects(
            existing_anns, classes, canvas_w, canvas_h,
        )
        initial_drawing = {"version": "4.4.0", "objects": initial_rects} if initial_rects else None

        canvas_result = st_canvas(
            fill_color=f"{stroke_color}33",
            stroke_width=2,
            stroke_color=stroke_color,
            background_image=pil_img.resize((canvas_w, canvas_h)),
            drawing_mode="rect",
            height=canvas_h,
            width=canvas_w,
            initial_drawing=initial_drawing,
            key=f"canvas_{selected_idx}",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Annotations", type="primary"):
                if canvas_result and canvas_result.json_data:
                    new_anns = _canvas_rects_to_annotations(
                        canvas_result.json_data,
                        class_name_to_id,
                        selected_class,
                        canvas_w,
                        canvas_h,
                    )
                    ds.update_annotations(img_path.name, new_anns)
                    st.success(f"Saved {len(new_anns)} annotations for {img_path.name}")
        with col2:
            if st.button("Clear Annotations"):
                ds.update_annotations(img_path.name, [])
                st.rerun()


# ===================================================================
# PAGE 3: Train
# ===================================================================

def _page_train(args: argparse.Namespace) -> None:
    st.header("Train")

    pdir = st.session_state.get("project_dir")
    classes = st.session_state.get("classes")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir or not classes:
        st.warning("Please create or load a project first (Setup page).")
        return

    ds = YOLODataset(project_dir=Path(pdir), classes=classes)

    # --- Hardware info ---
    hw = YOLOTrainer.detect_hardware()
    st.info(f"**Device:** {hw.display}")

    # --- Training config ---
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider(
            "Epochs",
            min_value=1, max_value=300, value=50,
            help="More epochs = longer training but potentially better accuracy. "
                 "50 is a good starting point.",
        )
    with col2:
        batch = st.number_input(
            "Batch size",
            min_value=1, max_value=128,
            value=hw.suggested_batch,
            help=f"Suggested: {hw.suggested_batch} based on your hardware.",
        )
    with col3:
        imgsz = st.selectbox(
            "Image size",
            options=[416, 640],
            index=1,
            help="640 is recommended for best accuracy.",
        )

    # --- Dataset split + YAML ---
    report = ds.quality_report()
    if report.total_images == 0:
        st.warning("No images in dataset. Upload and label images first.")
        return

    if report.annotated_images < 5:
        st.warning(
            f"Only {report.annotated_images} annotated images. "
            "At least 5 recommended before training."
        )

    # --- Training controls ---
    training_active = st.session_state.get("training_active", False)
    progress_state: TrainProgress = st.session_state.get(
        "train_progress", TrainProgress()
    )

    if not training_active:
        if st.button("Start Training", type="primary"):
            # Split and generate YAML
            ds.split()
            yaml_path = ds.generate_yaml()

            trainer = YOLOTrainer(
                base_model=base_model,
                project_dir=Path(pdir),
                device=hw.device,
            )

            st.session_state["training_active"] = True
            st.session_state["train_progress"] = TrainProgress(total_epochs=epochs)
            st.session_state["train_result"] = None

            def _run_training() -> None:
                def _cb(p: TrainProgress) -> None:
                    st.session_state["train_progress"] = p

                try:
                    result = trainer.train(
                        dataset_yaml=yaml_path,
                        epochs=epochs,
                        batch=batch,
                        imgsz=imgsz,
                        progress_callback=_cb,
                    )
                    st.session_state["train_result"] = result
                except Exception as exc:
                    st.session_state["train_progress"] = TrainProgress(
                        finished=True, error=str(exc),
                    )
                finally:
                    st.session_state["training_active"] = False

            st.session_state["trainer"] = trainer
            t = threading.Thread(target=_run_training, daemon=True)
            t.start()
            st.rerun()
    else:
        # Training in progress
        if st.button("Stop Training"):
            trainer = st.session_state.get("trainer")
            if trainer:
                trainer.request_stop()

        p = st.session_state.get("train_progress", TrainProgress())
        if p.total_epochs > 0:
            st.progress(
                p.epoch / p.total_epochs,
                text=p.message or f"Epoch {p.epoch}/{p.total_epochs}",
            )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")
        col2.metric("Box Loss", f"{p.box_loss:.4f}")
        col3.metric("Cls Loss", f"{p.cls_loss:.4f}")
        col4.metric("mAP50", f"{p.mAP50:.3f}")

        if p.error:
            st.error(f"Training error: {p.error}")
        elif not p.finished:
            # Auto-refresh every 3 seconds
            import time
            time.sleep(3)
            st.rerun()

    # --- Post-training summary ---
    result: TrainResult | None = st.session_state.get("train_result")
    if result:
        st.divider()
        st.subheader("Training Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final mAP50", f"{result.final_mAP50:.3f}")
        col2.metric("Final mAP50-95", f"{result.final_mAP50_95:.3f}")
        col3.metric("Model Size", f"{result.model_size_mb:.1f} MB")
        minutes = result.elapsed_seconds / 60
        col4.metric("Time", f"{minutes:.1f} min")

        if result.best_model:
            st.success(f"Best model saved to: `{result.best_model}`")


# ===================================================================
# PAGE 4: Evaluate & Deploy
# ===================================================================

def _page_evaluate_deploy(args: argparse.Namespace) -> None:
    st.header("Evaluate & Deploy")

    pdir = st.session_state.get("project_dir")
    classes = st.session_state.get("classes")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir or not classes:
        st.warning("Please create or load a project first (Setup page).")
        return

    pdir = Path(pdir)
    best_pt = pdir / "runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        st.warning("No trained model found. Complete training first.")
        return

    trainer = YOLOTrainer(base_model=base_model, project_dir=pdir)

    # --- Test image upload ---
    st.subheader("Test Your Model")
    test_files = st.file_uploader(
        "Upload test images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="test_upload",
    )

    if test_files:
        for f in test_files:
            img_bytes = f.read()
            pil_img = Image.open(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False))
            pil_img = Image.open(f).convert("RGB")
            img_array = np.array(pil_img)

            # Fine-tuned model
            try:
                detections = trainer.evaluate(img_array[..., ::-1])  # RGB->BGR
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                continue

            # Draw detections on image
            annotated = img_array.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['confidence']:.0%}"
                # Simple rectangle drawing with PIL
                from PIL import ImageDraw, ImageFont
                draw_img = Image.fromarray(annotated)
                draw = ImageDraw.Draw(draw_img)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x1, max(0, y1 - 15)), label, fill="green")
                annotated = np.array(draw_img)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original", use_container_width=True)
            with col2:
                st.image(annotated, caption="Fine-tuned model", use_container_width=True)

            # Detection summary
            if detections:
                summary = ", ".join(
                    f"{d['label']} ({d['confidence']:.0%})" for d in detections
                )
                st.write(f"Detections: {summary}")
            else:
                st.write("No detections.")

    # --- Quality assessment ---
    result: TrainResult | None = st.session_state.get("train_result")
    if result:
        st.divider()
        st.subheader("Quality Assessment")
        mAP = result.final_mAP50
        if mAP >= 0.7:
            st.success(
                f"Your model achieves **{mAP:.1%} mAP50** — this is good! "
                "It should reliably detect your custom objects."
            )
        elif mAP >= 0.4:
            st.warning(
                f"Your model achieves **{mAP:.1%} mAP50** — decent but could improve. "
                "Consider adding more training images, especially for underperforming classes."
            )
        else:
            st.error(
                f"Your model achieves **{mAP:.1%} mAP50** — this needs improvement. "
                "Add more images (50+ per class), ensure annotations are accurate, "
                "and try training for more epochs."
            )

    # --- Deploy ---
    st.divider()
    st.subheader("Deploy")

    deploy_dir = st.text_input(
        "Export directory",
        value=str(Path(args.base_path) / "custom_finetune"),
        help="Where to copy the ONNX model for use with pyzm detection.",
    )

    if st.button("Export ONNX", type="primary"):
        with st.spinner("Exporting to ONNX format..."):
            try:
                onnx_path = trainer.export_onnx(output_dir=Path(deploy_dir))
                st.success(f"ONNX model exported to: `{onnx_path}`")

                # Show config snippet
                st.subheader("pyzm Configuration")
                model_name = onnx_path.stem
                config_snippet = (
                    f"# Add to your detection config:\n"
                    f"models:\n"
                    f"  - name: {model_name}\n"
                    f"    type: object\n"
                    f"    framework: opencv\n"
                    f"    weights: {onnx_path}\n"
                    f"    min_confidence: 0.3\n"
                    f"    pattern: \"({'|'.join(classes)})\"\n"
                )
                st.code(config_snippet, language="yaml")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

    # --- Retrain shortcut ---
    st.divider()
    if st.button("Add More Images & Retrain"):
        st.session_state["selected_page"] = "Upload & Label"
        st.rerun()


# ===================================================================
# Main app
# ===================================================================

def main() -> None:
    st.set_page_config(
        page_title="pyzm YOLO Training",
        page_icon="🎯",
        layout="wide",
    )
    st.title("pyzm YOLO Fine-Tuning")

    args = _parse_app_args()
    project_root = Path(args.project_dir) if args.project_dir else DEFAULT_PROJECT_ROOT
    project_root.mkdir(parents=True, exist_ok=True)

    # Sidebar navigation
    pages = ["Setup", "Upload & Label", "Train", "Evaluate & Deploy"]
    default_idx = pages.index(st.session_state.get("selected_page", "Setup")) if st.session_state.get("selected_page") in pages else 0
    page = st.sidebar.radio("Navigation", pages, index=default_idx)
    st.session_state["selected_page"] = page

    # Show project info in sidebar
    if st.session_state.get("project_name"):
        st.sidebar.divider()
        st.sidebar.write(f"**Project:** {st.session_state['project_name']}")
        if st.session_state.get("classes"):
            st.sidebar.write(f"**Classes:** {', '.join(st.session_state['classes'])}")
        st.sidebar.write(f"**Base model:** {st.session_state.get('base_model', 'yolo11s')}")

    # Route to page
    if page == "Setup":
        _page_setup(args, project_root)
    elif page == "Upload & Label":
        _page_upload_label(args)
    elif page == "Train":
        _page_train(args)
    elif page == "Evaluate & Deploy":
        _page_evaluate_deploy(args)


if __name__ == "__main__":
    main()
