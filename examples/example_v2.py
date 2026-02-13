#!/usr/bin/env python3
"""pyzm v2 quick-start examples.

Each section is self-contained.  Comment/uncomment what you need.
"""

import pyzm
print(f"pyzm {pyzm.__version__}")


# ============================================================================
# 1. ZM API CLIENT (no ML needed)
# ============================================================================

from pyzm import ZMClient

zm = ZMClient(
    url="https://demo.zoneminder.com/zm/api",
    user="zmuser",
    password="zmpass",
    # verify_ssl=False,  # for self-signed certs
)

print(f"ZM {zm.zm_version}, API {zm.api_version}")

# -- Monitors --
for m in zm.monitors():
    print(f"  Monitor {m.id}: {m.name} ({m.function}) {m.width}x{m.height}")

# -- Events (last hour) --
events = zm.events(since="1 hour ago", limit=5)
for ev in events:
    print(f"  Event {ev.id}: {ev.cause} ({ev.length:.1f}s, {ev.alarm_frames} alarm frames)")

# -- Single event --
if events:
    ev = zm.event(events[0].id)
    print(f"  Event detail: {ev.name} notes={ev.notes!r}")

# -- Zones for a monitor --
if zm.monitors():
    zones = zm.monitor_zones(zm.monitors()[0].id)
    for z in zones:
        print(f"  Zone: {z.name} ({len(z.points)} points)")


# ============================================================================
# 2. ML DETECTION (no ZM needed)
# ============================================================================

from pyzm import Detector

# Quick start -- model names are resolved against base_path on disk
detector = Detector(models=["yolov4"])

# Or with explicit config:
#
# from pyzm import DetectorConfig, ModelConfig
# from pyzm.models.config import ModelFramework, ModelType, Processor
#
# detector = Detector(config=DetectorConfig(
#     models=[ModelConfig(
#         type=ModelType.OBJECT,
#         framework=ModelFramework.OPENCV,
#         processor=Processor.GPU,
#         weights="/path/to/yolov4.weights",
#         config="/path/to/yolov4.cfg",
#         labels="/path/to/coco.names",
#         min_confidence=0.5,
#     )],
# ))

# Detect on a local image
result = detector.detect("/tmp/image.jpg")

if result.matched:
    print(f"Detections: {result.summary}")
    # e.g. "person:97% car:85%"

    for det in result.detections:
        print(f"  {det.label}: {det.confidence:.0%} at ({det.bbox.x1},{det.bbox.y1})-({det.bbox.x2},{det.bbox.y2})")

    # Draw boxes and save
    annotated = result.annotate()
    import cv2
    cv2.imwrite("/tmp/detected.jpg", annotated)
else:
    print("No detections")


# ============================================================================
# 3. ZM + ML TOGETHER (detect on a ZM event)
# ============================================================================

from pyzm import ZMClient, Detector, StreamConfig
from pyzm.models.config import DetectorConfig

zm = ZMClient(url="https://zm.example.com/zm/api", user="admin", password="secret")
detector = Detector(models=["yolov4"])

event_id = 12345
zones = zm.monitor_zones(1)

result = detector.detect_event(
    zm,
    event_id,
    zones=zones,
    stream_config=StreamConfig(
        frame_set=["snapshot", "alarm", "1"],
        resize=800,
    ),
)

if result.matched:
    print(result.summary)
    result.annotate()  # draw boxes on the image

    # Update ZM event notes
    zm.update_event_notes(event_id, result.summary)


# ============================================================================
# 4. LOAD FROM YAML CONFIG (ml_sequence dict)
# ============================================================================

# If you already have an ml_sequence dict from your YAML config:
ml_sequence = {
    "general": {
        "model_sequence": "object,face",
    },
    "object": {
        "general": {"pattern": "(person|car|dog)", "same_model_sequence_strategy": "first"},
        "sequence": [
            {
                "object_framework": "coral_edgetpu",
                "object_weights": "/path/to/model.tflite",
                "object_labels": "/path/to/labels.txt",
                "object_min_confidence": 0.3,
            },
            {
                "object_framework": "opencv",
                "object_weights": "/path/to/yolov4.weights",
                "object_config": "/path/to/yolov4.cfg",
                "object_labels": "/path/to/coco.names",
                "object_processor": "gpu",
                "object_min_confidence": 0.5,
            },
        ],
    },
    "face": {
        "general": {"same_model_sequence_strategy": "first"},
        "sequence": [
            {
                "face_detection_framework": "dlib",
                "known_images_path": "/var/lib/zmeventnotification/known_faces",
                "face_model": "cnn",
                "face_recog_dist_threshold": 0.6,
            },
        ],
    },
}

detector = Detector.from_dict(ml_sequence)
result = detector.detect("/tmp/image.jpg")


# ============================================================================
# 5. LOGGING
# ============================================================================

from pyzm.log import setup_logging

# Console debug logging
logger = setup_logging(debug=True, component="myapp")
logger.info("Hello from pyzm")
logger.debug("Detail", extra={"zm_debug_level": 3})

# With ZM database logging (optional)
# logger = setup_logging(
#     debug=True,
#     component="myapp",
#     zm_db_host="localhost",
#     zm_db_user="zmuser",
#     zm_db_password="zmpass",
#     zm_db_name="zm",
# )
