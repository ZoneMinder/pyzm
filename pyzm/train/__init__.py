"""pyzm.train -- YOLO fine-tuning training module.

Provides a guided Streamlit web UI for fine-tuning YOLO models on custom
images so ZoneMinder users can detect custom objects alongside COCO classes.

Install::

    pip install pyzm[train]

Run::

    python -m pyzm.train
"""

from __future__ import annotations

__all__: list[str] = []


def check_dependencies() -> None:
    """Raise ImportError if required extras are missing."""
    missing: list[str] = []

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        missing.append("ultralytics>=8.3")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        missing.append("streamlit>=1.38")

    try:
        import streamlit_drawable_canvas  # noqa: F401
    except ImportError:
        missing.append("streamlit-drawable-canvas>=0.9")

    if missing:
        raise ImportError(
            "pyzm.train requires extra dependencies. Install with:\n\n"
            "  pip install pyzm[train]\n\n"
            f"Missing: {', '.join(missing)}"
        )
