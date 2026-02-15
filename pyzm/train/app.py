"""Streamlit UI for YOLO fine-tuning.

Launch with::

    python -m pyzm.train
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

_COLOR_PALETTE = [
    "#27AE60", "#8E44AD", "#0081FE", "#FE3C71",
    "#F38630", "#5BB12F", "#E74C3C", "#3498DB",
]


# ---------------------------------------------------------------------------
# CLI args (passed after -- by __main__.py)
# ---------------------------------------------------------------------------

def _parse_app_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--processor", default="gpu")
    args, _ = ap.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------

def _scan_models(base_path: str) -> list[dict]:
    """Return list of {name, path} for each .onnx/.pt file in base_path."""
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
    """Read class names from an ONNX model's metadata, or return empty."""
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


def _project_dir(project_root: Path, name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return project_root / safe


def _load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------

def _annotations_to_canvas_rects(
    annotations: list[Annotation],
    classes: list[str],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Convert YOLO annotations to fabric.js rect objects."""
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
            "left": x, "top": y, "width": w, "height": h,
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
        cls_name = obj.get("label", selected_class)
        cls_id = class_name_to_id.get(cls_name, class_name_to_id.get(selected_class, 0))
        left = obj["left"]
        top = obj["top"]
        w = obj["width"] * obj.get("scaleX", 1)
        h = obj["height"] * obj.get("scaleY", 1)
        cx = max(0.0, min(1.0, (left + w / 2) / img_w))
        cy = max(0.0, min(1.0, (top + h / 2) / img_h))
        nw = max(0.0, min(1.0, w / img_w))
        nh = max(0.0, min(1.0, h / img_h))
        annotations.append(Annotation(class_id=cls_id, cx=cx, cy=cy, w=nw, h=nh))
    return annotations


# ===================================================================
# STEP 1: Project configuration
# ===================================================================

def _section_config(args: argparse.Namespace, project_root: Path) -> None:
    """Project setup -- model selection, class groups, project name."""

    # Load existing projects
    existing = []
    if project_root.exists():
        existing = [
            d.name for d in sorted(project_root.iterdir())
            if d.is_dir() and (d / "project.json").exists()
        ]

    col_new, col_load = st.columns([3, 1])
    with col_new:
        project_name = st.text_input(
            "Project name",
            value=st.session_state.get("project_name", ""),
            placeholder="e.g. my-packages",
        )
    with col_load:
        if existing:
            st.write("")  # spacing
            selected = st.selectbox("or load existing", [""] + existing, label_visibility="collapsed")
            if selected:
                pdir = project_root / selected
                ds = YOLODataset.load(pdir)
                st.session_state["project_name"] = selected
                st.session_state["project_dir"] = str(pdir)
                st.session_state["classes"] = ds.classes
                st.session_state["class_groups"] = ds.class_groups
                meta = json.loads((pdir / "project.json").read_text())
                st.session_state["base_model"] = meta.get("base_model", "yolo11s")
                st.rerun()

    # --- Model selection ---
    available = _scan_models(args.base_path)
    model_names = [m["name"] for m in available]
    model_paths = {m["name"]: m["path"] for m in available}

    default_idx = 0
    for i, name in enumerate(model_names):
        if name == "yolo11s":
            default_idx = i
            break

    base_model = st.selectbox(
        "Base model",
        options=model_names,
        index=default_idx,
        format_func=lambda n: f"{n}  ({model_paths[n]})",
    )

    # --- Class grouping ---
    model_path = model_paths.get(base_model, "")
    model_classes = _read_model_classes(model_path)

    if model_classes:
        st.caption(
            f"This model knows {len(model_classes)} classes. "
            "Define your output classes below. You can group multiple "
            "model classes into one (e.g. car + truck + bus -> vehicle)."
        )
    else:
        st.caption("Could not read classes from model. Define classes manually.")

    # Initialise groups from session or defaults
    if "class_groups_edit" not in st.session_state:
        saved = st.session_state.get("class_groups", {})
        if saved:
            st.session_state["class_groups_edit"] = [
                {"name": name, "sources": srcs}
                for name, srcs in saved.items()
            ]
        else:
            # Sensible defaults
            st.session_state["class_groups_edit"] = [
                {"name": "person", "sources": ["person"]},
            ]

    groups = st.session_state["class_groups_edit"]

    # Render each group row
    to_delete = None
    used_sources: set[str] = set()
    for g in groups:
        used_sources.update(g["sources"])

    for idx, group in enumerate(groups):
        col_name, col_sources, col_del = st.columns([2, 5, 1])
        with col_name:
            new_name = st.text_input(
                "Class name", value=group["name"],
                key=f"grp_name_{idx}",
                label_visibility="collapsed",
                placeholder="e.g. vehicle",
            )
            group["name"] = new_name.strip()
        with col_sources:
            if model_classes:
                # Show model classes available for this group
                # Include currently assigned + anything not used by other groups
                other_used = set()
                for j, g in enumerate(groups):
                    if j != idx:
                        other_used.update(g["sources"])
                available_for_group = [
                    c for c in model_classes
                    if c not in other_used or c in group["sources"]
                ]
                group["sources"] = st.multiselect(
                    "Model classes",
                    options=available_for_group,
                    default=[s for s in group["sources"] if s in available_for_group],
                    key=f"grp_src_{idx}",
                    label_visibility="collapsed",
                )
            else:
                src_text = st.text_input(
                    "Source classes (comma-sep)",
                    value=", ".join(group["sources"]),
                    key=f"grp_src_{idx}",
                    label_visibility="collapsed",
                )
                group["sources"] = [s.strip() for s in src_text.split(",") if s.strip()]
        with col_del:
            if st.button("X", key=f"grp_del_{idx}"):
                to_delete = idx

    if to_delete is not None:
        groups.pop(to_delete)
        st.rerun()

    if st.button("+ Add class"):
        groups.append({"name": "", "sources": []})
        st.rerun()

    # Custom classes (no model source -- purely new)
    custom_input = st.text_input(
        "Custom classes with no model equivalent (comma-separated)",
        value=st.session_state.get("custom_classes_input", ""),
        placeholder="e.g. package, my_pet",
    )
    custom_classes = [c.strip() for c in custom_input.split(",") if c.strip()]

    # Build final class list and groups dict
    class_groups_dict: dict[str, list[str]] = {}
    final_classes: list[str] = []
    for g in groups:
        name = g["name"]
        if not name:
            continue
        final_classes.append(name)
        class_groups_dict[name] = g["sources"]
    for c in custom_classes:
        if c not in final_classes:
            final_classes.append(c)
            class_groups_dict[c] = []  # no source mapping

    if final_classes:
        parts = []
        for cls in final_classes:
            srcs = class_groups_dict.get(cls, [])
            if srcs and srcs != [cls]:
                parts.append(f"**{cls}** ({', '.join(srcs)})")
            else:
                parts.append(f"**{cls}**")
        st.caption("Output classes: " + " | ".join(parts))

    if st.button("Create / Update Project", type="primary", disabled=not project_name.strip()):
        if not final_classes:
            st.error("Define at least one class.")
            return

        pdir = _project_dir(project_root, project_name)
        ds = YOLODataset(project_dir=pdir, classes=final_classes, class_groups=class_groups_dict)
        ds.init_project()

        # Save base_model into project.json
        meta_path = pdir / "project.json"
        meta = json.loads(meta_path.read_text())
        meta["base_model"] = base_model
        meta_path.write_text(json.dumps(meta, indent=2))

        st.session_state["project_name"] = project_name
        st.session_state["project_dir"] = str(pdir)
        st.session_state["classes"] = final_classes
        st.session_state["class_groups"] = class_groups_dict
        st.session_state["custom_classes_input"] = custom_input
        st.session_state["base_model"] = base_model
        st.rerun()


# ===================================================================
# STEP 2: Upload images
# ===================================================================

def _section_upload() -> YOLODataset | None:
    """Upload images and return the dataset, or None if no project."""
    pdir = st.session_state.get("project_dir")
    classes = st.session_state.get("classes")
    if not pdir or not classes:
        st.info("Create or load a project above first.")
        return None

    ds = YOLODataset(project_dir=Path(pdir), classes=classes)
    uploaded = st.file_uploader(
        "Drag and drop images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        added = 0
        for f in uploaded:
            tmp = Path(tempfile.mkdtemp()) / f.name
            tmp.write_bytes(f.read())
            ds.add_image(tmp, [])
            added += 1
        st.toast(f"Added {added} images")
        st.rerun()

    images = ds.staged_images()
    if images:
        st.caption(f"{len(images)} images in project")
    return ds


# ===================================================================
# STEP 3: Auto-detect
# ===================================================================

def _section_auto_detect(ds: YOLODataset, args: argparse.Namespace) -> None:
    """Run the selected model on all images to pre-label."""
    images = ds.staged_images()
    if not images:
        return

    base_model = st.session_state.get("base_model", "yolo11s")
    classes = st.session_state.get("classes", [])
    class_groups = st.session_state.get("class_groups", {})

    # Count how many are already annotated
    annotated = sum(1 for img in images if ds.annotations_for(img.name))
    unannotated = len(images) - annotated
    if annotated:
        st.caption(f"{annotated} labeled, {unannotated} unlabeled")

    # Show mapping summary if groups are non-trivial
    non_trivial = {k: v for k, v in class_groups.items() if v and v != [k]}
    if non_trivial:
        mapping_parts = [f"{', '.join(srcs)} -> **{name}**" for name, srcs in non_trivial.items()]
        st.caption("Class mapping: " + " | ".join(mapping_parts))

    if st.button(f"Auto-detect using {base_model}"):
        progress_bar = st.progress(0, text=f"Running {base_model} on {len(images)} images...")
        try:
            from pyzm.train.auto_label import auto_label, build_class_mapping

            class_mapping = build_class_mapping(class_groups) if class_groups else None

            results = auto_label(
                image_paths=images,
                model_name=base_model,
                base_path=args.base_path,
                processor=args.processor,
                target_classes=classes,
                class_mapping=class_mapping,
            )
            total_anns = 0
            for img_path, anns in results.items():
                if anns:
                    ds.update_annotations(img_path.name, anns)
                    total_anns += len(anns)
            progress_bar.progress(1.0, text="Done")
            st.toast(f"Auto-detected {total_anns} objects across {len(images)} images")
            st.rerun()
        except Exception as exc:
            st.error(f"Auto-detection failed: {exc}")


# ===================================================================
# STEP 4: Label / review annotations
# ===================================================================

def _section_label(ds: YOLODataset) -> None:
    """Image gallery + annotation canvas."""
    images = ds.staged_images()
    if not images:
        return

    classes = st.session_state.get("classes", [])
    class_name_to_id = {c: i for i, c in enumerate(classes)}

    # --- Quality summary in sidebar ---
    with st.sidebar:
        st.markdown("### Dataset")
        report = ds.quality_report()
        st.write(f"Images: **{report.total_images}** ({report.annotated_images} labeled)")
        for cls, count in report.per_class.items():
            color = _COLOR_PALETTE[classes.index(cls) % len(_COLOR_PALETTE)] if cls in classes else "#999"
            st.write(f"<span style='color:{color}'>&#9632;</span> {cls}: **{count}**", unsafe_allow_html=True)
        for w in report.warnings:
            st.warning(w.message)

    # --- Thumbnail gallery ---
    selected_idx = st.session_state.get("selected_image_idx", 0)
    cols_per_row = 6
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
                thumb.thumbnail((100, 100))
                border = "2px solid #27AE60" if anns else "2px solid #555"
                if idx == selected_idx:
                    border = "3px solid #0081FE"
                st.markdown(
                    f"<div style='border:{border}; border-radius:4px; padding:2px; cursor:pointer;'>",
                    unsafe_allow_html=True,
                )
                st.image(thumb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                label_text = f"{len(anns)}" if anns else "-"
                if st.button(label_text, key=f"sel_{idx}", use_container_width=True):
                    st.session_state["selected_image_idx"] = idx
                    st.rerun()

    # --- Annotation canvas for selected image ---
    if selected_idx >= len(images):
        selected_idx = 0
        st.session_state["selected_image_idx"] = 0

    img_path = images[selected_idx]
    pil_img = _load_image_pil(img_path)
    img_w, img_h = pil_img.size

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("Prev", disabled=selected_idx == 0, use_container_width=True):
            st.session_state["selected_image_idx"] = selected_idx - 1
            st.rerun()
    with nav_col2:
        st.caption(f"{img_path.name}  ({selected_idx + 1}/{len(images)})")
    with nav_col3:
        if st.button("Next", disabled=selected_idx >= len(images) - 1, use_container_width=True):
            st.session_state["selected_image_idx"] = selected_idx + 1
            st.rerun()

    # Class selector for new boxes
    selected_class = st.selectbox(
        "Class for new boxes",
        options=classes,
        key="annotation_class",
    )
    color_idx = class_name_to_id[selected_class]
    stroke_color = _COLOR_PALETTE[color_idx % len(_COLOR_PALETTE)]

    # Scale canvas
    scale = min(1.0, 700 / img_w)
    canvas_w = int(img_w * scale)
    canvas_h = int(img_h * scale)

    existing_anns = ds.annotations_for(img_path.name)
    initial_rects = _annotations_to_canvas_rects(existing_anns, classes, canvas_w, canvas_h)
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

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Save", type="primary", use_container_width=True):
            if canvas_result and canvas_result.json_data:
                new_anns = _canvas_rects_to_annotations(
                    canvas_result.json_data, class_name_to_id,
                    selected_class, canvas_w, canvas_h,
                )
                ds.update_annotations(img_path.name, new_anns)
                st.toast(f"Saved {len(new_anns)} annotations")
    with btn_col2:
        if st.button("Clear", use_container_width=True):
            ds.update_annotations(img_path.name, [])
            st.rerun()


# ===================================================================
# STEP 5: Train
# ===================================================================

def _section_train(ds: YOLODataset, args: argparse.Namespace) -> None:
    """Training controls and live progress."""
    pdir = st.session_state.get("project_dir")
    classes = st.session_state.get("classes")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir or not classes:
        return

    report = ds.quality_report()
    if report.annotated_images < 2:
        st.caption("Need at least 2 annotated images to train.")
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
        st.text("")  # spacing
        st.caption(hw.display)

    training_active = st.session_state.get("training_active", False)

    if not training_active:
        if st.button("Start Training", type="primary"):
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
            st.session_state["trainer"] = trainer

            def _run() -> None:
                def _cb(p: TrainProgress) -> None:
                    st.session_state["train_progress"] = p
                try:
                    result = trainer.train(
                        dataset_yaml=yaml_path, epochs=epochs,
                        batch=batch, imgsz=imgsz, progress_callback=_cb,
                    )
                    st.session_state["train_result"] = result
                except Exception as exc:
                    st.session_state["train_progress"] = TrainProgress(finished=True, error=str(exc))
                finally:
                    st.session_state["training_active"] = False

            threading.Thread(target=_run, daemon=True).start()
            st.rerun()
    else:
        # Live progress
        p: TrainProgress = st.session_state.get("train_progress", TrainProgress())
        if p.total_epochs > 0:
            st.progress(p.epoch / p.total_epochs, text=p.message or f"Epoch {p.epoch}/{p.total_epochs}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Epoch", f"{p.epoch}/{p.total_epochs}")
        m2.metric("Box Loss", f"{p.box_loss:.4f}")
        m3.metric("Cls Loss", f"{p.cls_loss:.4f}")
        m4.metric("mAP50", f"{p.mAP50:.3f}")

        if st.button("Stop"):
            trainer = st.session_state.get("trainer")
            if trainer:
                trainer.request_stop()

        if p.error:
            st.error(p.error)
        elif not p.finished:
            import time
            time.sleep(2)
            st.rerun()

    # Results
    result: TrainResult | None = st.session_state.get("train_result")
    if result:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("mAP50", f"{result.final_mAP50:.3f}")
        r2.metric("mAP50-95", f"{result.final_mAP50_95:.3f}")
        r3.metric("Size", f"{result.model_size_mb:.1f} MB")
        r4.metric("Time", f"{result.elapsed_seconds / 60:.1f} min")
        if result.best_model:
            st.code(str(result.best_model), language=None)


# ===================================================================
# STEP 6: Export
# ===================================================================

def _section_export(args: argparse.Namespace) -> None:
    """ONNX export and deployment config."""
    pdir = st.session_state.get("project_dir")
    classes = st.session_state.get("classes")
    base_model = st.session_state.get("base_model", "yolo11s")
    if not pdir:
        return

    best_pt = Path(pdir) / "runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        return

    deploy_dir = st.text_input(
        "Export to",
        value=str(Path(args.base_path) / "custom_finetune"),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export ONNX", type="primary"):
            trainer = YOLOTrainer(base_model=base_model, project_dir=Path(pdir))
            with st.spinner("Exporting..."):
                try:
                    onnx_path = trainer.export_onnx(output_dir=Path(deploy_dir))
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

    # Quick test
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
                st.image(draw_img, use_container_width=True)
                if dets:
                    st.caption(", ".join(f"{d['label']} {d['confidence']:.0%}" for d in dets))
                else:
                    st.caption("No detections")
            except Exception as exc:
                st.error(str(exc))


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    st.set_page_config(page_title="pyZM: Customize your own ML model", layout="wide")

    args = _parse_app_args()
    project_root = Path(args.project_dir) if args.project_dir else DEFAULT_PROJECT_ROOT
    project_root.mkdir(parents=True, exist_ok=True)

    st.markdown("## pyZM: Customize your own ML model")

    # --- Step 1: Config ---
    with st.expander("1. Project", expanded=not st.session_state.get("project_dir")):
        _section_config(args, project_root)

    if not st.session_state.get("project_dir"):
        return

    ds = YOLODataset(
        project_dir=Path(st.session_state["project_dir"]),
        classes=st.session_state.get("classes", []),
    )

    # --- Step 2: Upload ---
    with st.expander("2. Upload Images", expanded=True):
        _section_upload()

    # --- Step 3: Auto-detect ---
    if ds.staged_images():
        with st.expander("3. Auto-Detect", expanded=True):
            _section_auto_detect(ds, args)

        # --- Step 4: Label ---
        with st.expander("4. Review & Label", expanded=True):
            _section_label(ds)

        # --- Step 5: Train ---
        with st.expander("5. Train", expanded=bool(st.session_state.get("training_active") or st.session_state.get("train_result"))):
            _section_train(ds, args)

        # --- Step 6: Export ---
        best_pt = Path(st.session_state["project_dir"]) / "runs" / "train" / "weights" / "best.pt"
        if best_pt.exists() or st.session_state.get("train_result"):
            with st.expander("6. Export & Test", expanded=bool(st.session_state.get("train_result"))):
                _section_export(args)


if __name__ == "__main__":
    main()
