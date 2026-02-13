Testing
========

pyzm has two tiers of tests: unit/integration tests that need no special
hardware, and end-to-end (e2e) tests that run real ML models against real
images.

Running all tests
------------------

.. code-block:: bash

   pip install pytest
   python -m pytest tests/ -v

This runs both unit and e2e tests.  The e2e tests are automatically
skipped if the model directory or test image are missing.


Unit / integration tests
-------------------------

These tests mock backends and use no real models.  They run anywhere:

.. code-block:: bash

   python -m pytest tests/ -m "not e2e" -v


End-to-end tests
-----------------

The ``tests/test_e2e/`` directory contains 89 tests that exercise every
objectconfig feature using real YOLO models and a real test image
(``tests/test_e2e/bird.jpg``, included in the repository).

**Prerequisites:**

- ML models installed at ``/var/lib/zmeventnotification/models/``
  (at least one YOLO model, e.g. ``yolov4/``)
- Python packages: ``opencv-python``, ``numpy``, ``shapely``
- For remote-serve tests: ``fastapi``, ``uvicorn``, ``requests``,
  ``python-jose``, ``passlib``

**Run all e2e tests:**

.. code-block:: bash

   python -m pytest tests/test_e2e/ -v

**Skip the slower remote-serve tests** (which start real server subprocesses):

.. code-block:: bash

   python -m pytest tests/test_e2e/ -v -m "not serve"

**Run only remote-serve tests:**

.. code-block:: bash

   python -m pytest tests/test_e2e/ -v -m serve

**Run a single test file:**

.. code-block:: bash

   python -m pytest tests/test_e2e/test_zone_filtering.py -v


Test file reference
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - File
     - Tests
     - What it covers
   * - ``test_basic_detection.py``
     - 5
     - File path input, numpy input, multi-frame, result properties, to_dict roundtrip
   * - ``test_model_discovery.py``
     - 5
     - Auto-discover all, directory name resolution, file stem resolution, unknown fallback, framework assignment
   * - ``test_pattern_filtering.py``
     - 4
     - Per-model pattern, global pattern, restrictive regex, specific label match
   * - ``test_size_filtering.py``
     - 5
     - ``max_detection_size`` as percentage, pixels, None, global level, large threshold
   * - ``test_zone_filtering.py``
     - 6
     - Full-image zone, tiny zone, zone-specific pattern, non-matching pattern, multiple zones, no zones
   * - ``test_min_confidence.py``
     - 2
     - High threshold filters low-confidence, low vs high comparison
   * - ``test_disabled_models.py``
     - 2
     - ``enabled=False`` skipped, mixed enabled/disabled
   * - ``test_match_strategies.py``
     - 4
     - ``FIRST``, ``MOST``, ``MOST_UNIQUE``, ``UNION``
   * - ``test_frame_strategies.py``
     - 4
     - ``FIRST``, ``MOST``, ``MOST_UNIQUE``, ``MOST_MODELS``
   * - ``test_pre_existing_labels.py``
     - 2
     - Gate not satisfied (skips), gate satisfied (runs)
   * - ``test_past_detections.py``
     - 6
     - First run, duplicate filtering, aliases, ``ignore_labels``, per-label area override, moved object
   * - ``test_from_dict.py``
     - 8
     - ``Detector.from_dict()`` with patterns, ``match_past_detections``, disabled, per-label area keys, aliases, ``ignore_past_detection_labels``, gateway settings
   * - ``test_stream_config.py``
     - 8
     - ``StreamConfig.from_dict()`` defaults, ``resize``, ``frame_set`` (string/list), bools, ints, unknown keys
   * - ``test_model_dimensions.py``
     - 2
     - Custom ``model_width``/``model_height``, default (None)
   * - ``test_multi_model.py``
     - 2
     - Two-model UNION, multiple model names by string
   * - ``test_pipeline_loading.py``
     - 3
     - Eager load (``is_loaded=True``), lazy prepare (``is_loaded=False``), lazy loads on first detect
   * - ``test_remote_serve.py``
     - 8
     - ``/health``, ``/detect``, ``/models`` (eager), ``--models all`` (lazy), gateway image mode, zones via serve, JWT auth flow, gateway with auth
   * - ``test_server_config.py``
     - 3
     - ``["all"]`` valid, ``["all", "yolov4"]`` raises, normal models valid
   * - ``test_filter_combinations.py``
     - 4
     - Pattern + size, zone + pattern, global pattern + zone, past detection + zone
   * - ``test_edge_cases.py``
     - 6
     - Bad file path, empty frames list, no models, ``filter_by_pattern()``, ``annotate()``, annotate with no image


Pytest markers
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - Description
   * - ``integration``
     - Tests requiring optional dependencies (cv2, shapely, numpy)
   * - ``e2e``
     - End-to-end tests requiring real models and images on disk
   * - ``serve``
     - Tests that start a ``pyzm.serve`` subprocess (slower, need fastapi/uvicorn)
