ML Detection Deep-Dive
======================

This guide covers the architecture and configuration of pyzm's ML
detection pipeline in detail.


Architecture overview
----------------------

.. code-block:: text

   Detector
     |
     +-- ModelPipeline
     |     |
     |     +-- YoloBackend        (OpenCV DNN / ONNX)
     |     +-- CoralBackend       (Coral EdgeTPU)
     |     +-- FaceDlibBackend    (dlib / face_recognition)
     |     +-- AlprBackend        (PlateRecognizer / OpenALPR)
     |     +-- RekognitionBackend (AWS Rekognition)
     |     |
     |     +-- filters (zone, size, pattern, past-detection)
     |
     +-- DetectorConfig
     +-- StreamConfig (for event-based detection)

``Detector`` is the public API. It owns a ``ModelPipeline`` that
sequences the configured backends, applies match strategies, and runs
post-detection filters.

When ``gateway`` is set, ``Detector`` skips local inference and sends
images to a remote ``pyzm.serve`` server instead.  With
``gateway_mode="url"``, ``detect_event()`` sends frame URLs to the
server and the server fetches images directly from ZoneMinder (see the
:doc:`serve guide </guide/serve>`).


Configuration
--------------

DetectorConfig
~~~~~~~~~~~~~~~

Top-level detection settings:

.. code-block:: python

   from pyzm.models.config import DetectorConfig, ModelConfig

   config = DetectorConfig(
       models=[...],                     # list of ModelConfig
       match_strategy="most",            # first | most | most_unique | union
       frame_strategy="most_models",     # first | most | most_unique | most_models
       pattern=".*",                     # global label regex filter
       max_detection_size="50%",         # max bbox size (% of image or "Npx")
       match_past_detections=False,      # compare with previous run
       past_det_max_diff_area="5%",      # area tolerance for past matching
   )

ModelConfig
~~~~~~~~~~~~

Per-model settings:

.. code-block:: python

   from pyzm.models.config import ModelConfig, ModelFramework, Processor

   model = ModelConfig(
       name="YoloV4",
       type="object",                    # object | face | alpr
       framework=ModelFramework.OPENCV,  # opencv | coral_edgetpu | face_dlib | ...
       processor=Processor.GPU,          # cpu | gpu | tpu
       weights="/path/to/yolov4.weights",
       config="/path/to/yolov4.cfg",
       labels="/path/to/coco.names",
       min_confidence=0.3,
       pattern="(person|car|dog)",
   )

See the API reference for the full list of fields (face-specific,
ALPR-specific, AWS, lock management, etc.).

StreamConfig
~~~~~~~~~~~~~

Controls frame extraction from ZM events:

.. code-block:: python

   from pyzm import StreamConfig

   stream_cfg = StreamConfig(
       frame_set=["snapshot", "alarm", "1"],  # which frames to fetch
       resize=800,               # resize longest edge to N pixels
       max_frames=0,             # 0 = no limit
       start_frame=1,            # first frame index
       frame_skip=1,             # skip every N frames
       max_attempts=3,           # retries on failure
       sleep_between_attempts=4, # seconds between retries
   )

``frame_set`` values: ``"snapshot"`` (the ZM snapshot image), ``"alarm"``
(alarm frames), or integer frame IDs as strings.


Model discovery
----------------

When you pass a string model name to ``Detector(models=["yolov4"])``,
pyzm resolves it by scanning ``base_path`` (default:
``/var/lib/zmeventnotification/models``):

.. code-block:: text

   /var/lib/zmeventnotification/models/
     yolov4/
       yolov4.weights
       yolov4.cfg
       coco.names
     coral_edgetpu/
       ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
       coco_indexed.names

The framework is auto-detected from the file extensions (``.weights`` +
``.cfg`` = OpenCV YOLO, ``.tflite`` = Coral EdgeTPU).


The ``ml_sequence`` dict format
--------------------------------

``objectconfig.yml`` uses a nested dict format that maps directly to
``DetectorConfig.from_dict()``:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object,face,alpr"

     object:
       general:
         pattern: "(person|car|dog)"
         same_model_sequence_strategy: "most"
       sequence:
         - name: "YoloV4"
           object_framework: opencv
           object_weights: /path/to/yolov4.weights
           object_config: /path/to/yolov4.cfg
           object_labels: /path/to/coco.names
           object_min_confidence: 0.3
           object_processor: cpu

     face:
       general:
         same_model_sequence_strategy: first
       sequence:
         - face_detection_framework: dlib
           known_images_path: /path/to/known_faces
           face_model: cnn

     alpr:
       general:
         same_model_sequence_strategy: first
         pre_existing_labels: ["car", "bus", "truck"]
       sequence:
         - alpr_service: plate_recognizer
           alpr_key: YOUR_KEY

In Python:

.. code-block:: python

   import yaml
   from pyzm import Detector

   with open("objectconfig.yml") as f:
       cfg = yaml.safe_load(f)

   detector = Detector.from_dict(cfg["ml_sequence"])


Supported backends
-------------------

All backends implement the ``MLBackend`` interface (``load()``,
``detect(image)``, ``name``).

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - Framework value
     - Description
   * - YoloBackend
     - ``opencv``
     - OpenCV DNN with Darknet (``.weights``) or ONNX models
   * - CoralBackend
     - ``coral_edgetpu``
     - Google Coral EdgeTPU via ``pycoral``
   * - FaceDlibBackend
     - ``face_dlib``
     - Face recognition using dlib / ``face_recognition``
   * - AlprBackend
     - ``plate_recognizer``, ``openalpr``
     - License plate recognition (cloud or local)
   * - RekognitionBackend
     - ``aws_rekognition``
     - AWS Rekognition API


Match and frame strategies
---------------------------

Frame strategies
~~~~~~~~~~~~~~~~~

When multiple frames are extracted from an event, the ``frame_strategy``
determines which frame's detections to return:

- **first** -- return the first frame that has any detections
- **most** -- return the frame with the most total detections
- **most_unique** -- return the frame with the most unique labels
- **most_models** -- return the frame where the most models contributed
  detections (default)

Match strategies
~~~~~~~~~~~~~~~~~

When multiple model variants are configured for the same type (e.g. two
object detectors), the ``match_strategy`` determines how their results
are combined:

- **first** -- use results from the first model that detects anything
- **most** -- use the model with the most detections (default)
- **most_unique** -- use the model with the most unique labels
- **union** -- merge all detections from all models


Zone-based filtering
---------------------

Zones are polygons that define regions of interest in the camera frame.
Only detections whose bounding box intersects a zone are kept.

.. code-block:: python

   from pyzm.models.zm import Zone

   zones = [
       Zone(name="driveway", points=[(0,300), (600,300), (600,480), (0,480)]),
       Zone(name="porch", points=[(200,0), (400,0), (400,200), (200,200)],
            pattern="person"),
   ]

   result = detector.detect("/path/to/image.jpg", zones=zones)

Each zone can have its own ``pattern`` regex. A detection must match the
zone's pattern *and* physically intersect the polygon to be kept.

When using ZoneMinder events, use ``zm.monitor_zones(monitor_id)`` to
fetch zones configured in the ZM web UI.


Past-detection filtering
-------------------------

When ``match_past_detections=True``, pyzm compares detections against a
pickled file from the previous run. Detections whose bounding box is in
roughly the same position (within ``past_det_max_diff_area``) are
filtered out. This prevents repeated notifications for parked cars,
stationary objects, etc.

.. code-block:: python

   config = DetectorConfig(
       models=[...],
       match_past_detections=True,
       past_det_max_diff_area="5%",
       past_det_max_diff_area_labels={"car": "10%"},
       ignore_past_detection_labels=["person"],
       aliases=[["car", "bus", "truck"]],
   )

- ``past_det_max_diff_area`` -- area difference tolerance (default ``"5%"``)
- ``past_det_max_diff_area_labels`` -- per-label overrides
- ``ignore_past_detection_labels`` -- labels to always keep (never filter)
- ``aliases`` -- treat these labels as equivalent when matching


Result objects
---------------

DetectionResult
~~~~~~~~~~~~~~~~

.. code-block:: python

   result.matched          # bool -- any detections?
   result.labels           # list[str]
   result.confidences      # list[float]
   result.boxes            # list[list[int]]  -- [x1,y1,x2,y2] per detection
   result.summary          # str -- "person:97% car:85%"
   result.frame_id         # int | str | None
   result.image            # np.ndarray | None (the source image)
   result.image_dimensions # dict

   result.annotate()       # draw boxes, return annotated image
   result.filter_by_pattern("person")  # new result with only matching labels
   result.to_dict()        # serialize to dict

Detection
~~~~~~~~~~

.. code-block:: python

   det.label           # str
   det.confidence      # float
   det.bbox            # BBox
   det.model_name      # str
   det.detection_type  # str ("object", "face", "alpr")

BBox
~~~~~

.. code-block:: python

   bbox.x1, bbox.y1   # top-left corner
   bbox.x2, bbox.y2   # bottom-right corner
   bbox.width          # x2 - x1
   bbox.height         # y2 - y1
   bbox.area           # width * height
   bbox.center         # (cx, cy) tuple
