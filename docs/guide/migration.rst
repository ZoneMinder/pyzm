Migrating from v1 to v2
========================

pyzm v2 is a major rewrite. This guide shows the before/after for every
significant change so you can update your code incrementally.


API client
-----------

**v1:**

.. code-block:: python

   import pyzm.api as zmapi

   api = zmapi.ZMApi(options={
       "apiurl": "https://zm.example.com/zm/api",
       "portalurl": "https://zm.example.com/zm",
       "user": "admin",
       "password": "secret",
   })

**v2:**

.. code-block:: python

   from pyzm import ZMClient

   zm = ZMClient(
       url="https://zm.example.com/zm/api",
       user="admin",
       password="secret",
   )

The ``portal_url`` is derived automatically from the API URL. Pass it
explicitly if your portal lives at a different host.


Monitors
~~~~~~~~~

**v1:**

.. code-block:: python

   ms = api.monitors()
   for m in ms.list():
       print(m.name())

**v2:**

.. code-block:: python

   for m in zm.monitors():
       print(m.name)  # dataclass attribute, not a method

``zm.monitors()`` returns ``list[Monitor]`` directly -- no ``.list()`` call,
no wrapper object. Fields are attributes (``m.id``, ``m.name``, ``m.function``,
``m.width``, ``m.height``).


Events
~~~~~~~

**v1:**

.. code-block:: python

   events = api.events(options={
       "from": "2024-01-01",
       "to": "2024-01-02",
       "object_only": True,
   })
   for e in events.list():
       print(e.name())

**v2:**

.. code-block:: python

   events = zm.events(since="1 day ago", limit=50)
   for e in events:
       print(e.name)

Human-readable time strings (``"1 hour ago"``, ``"yesterday"``) are
supported via ``dateparser``. Returns ``list[Event]`` directly.


Frame extraction
~~~~~~~~~~~~~~~~~

**v1:**

.. code-block:: python

   frames = event.download_image(...)

**v2:**

.. code-block:: python

   frames, dims = zm.get_event_frames(
       event_id=12345,
       stream_config=StreamConfig(frame_set=["snapshot", "alarm"], resize=800),
   )

Frame extraction is now handled by ``ZMClient.get_event_frames()`` using a
``StreamConfig`` object instead of ad-hoc keyword arguments.


Logging
--------

**v1:**

.. code-block:: python

   import pyzm.ZMLog as zmlog
   zmlog.init(name="myapp")
   zmlog.Info("Something happened")
   zmlog.Debug(3, "Verbose message")

**v2:**

.. code-block:: python

   from pyzm.log import setup_logging

   logger = setup_logging(debug=True, component="myapp")
   logger.info("Something happened")
   logger.debug("Verbose message", extra={"zm_debug_level": 3})

``setup_logging()`` returns a standard ``logging.Logger``. The old
``pyzm.ZMLog`` module is still importable for backward compatibility.


ML detection
-------------

**v1:** Direct backend usage:

.. code-block:: python

   from pyzm.ml.yolo import Yolo

   model = Yolo(options={...})
   model.load()
   results = model.detect(image)
   # results is a nested list: [[labels], [confidences], [boxes]]

**v2:** Unified ``Detector`` API:

.. code-block:: python

   from pyzm import Detector

   detector = Detector(models=["yolov4"])
   result = detector.detect("/path/to/image.jpg")
   # result is a DetectionResult with typed fields

You never import backend classes directly. The ``Detector`` creates and
manages the appropriate backend based on your ``ModelConfig``.


Detection results
~~~~~~~~~~~~~~~~~~

**v1:** Index into nested lists:

.. code-block:: python

   labels = matched_data["labels"]        # ["person", "car"]
   confs  = matched_data["confidences"]   # [0.97, 0.85]
   boxes  = matched_data["boxes"]         # [[x1,y1,x2,y2], ...]

   first_label = labels[0]

**v2:** Typed ``DetectionResult`` object:

.. code-block:: python

   result.labels          # ["person", "car"]
   result.confidences     # [0.97, 0.85]
   result.boxes           # [[x1,y1,x2,y2], ...]
   result.summary         # "person:97% car:85%"
   result.matched         # True

   det = result.detections[0]
   det.label              # "person"
   det.confidence         # 0.97
   det.bbox.x1            # 100
   det.bbox.area          # 50000

   # Draw boxes on the image
   annotated = result.annotate()


Configuration
~~~~~~~~~~~~~~

**v1:** ``objectconfig.ini`` with ``{{template}}`` substitution:

.. code-block:: ini

   [ml]
   ml_sequence=object,face
   object_detection_pattern=(person|car)
   ...

**v2:** ``objectconfig.yml`` (YAML, no templates):

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object,face"
     object:
       general:
         pattern: "(person|car)"
       sequence:
         - object_framework: opencv
           object_weights: /path/to/yolov4.weights
           ...

In Python, use ``Detector.from_dict(ml_options)`` to load the dict
directly.


Remote detection
~~~~~~~~~~~~~~~~~

**v1:** Separate ``mlapi`` daemon + ``remote_detect()``:

.. code-block:: python

   # Required running mlapi as a separate service
   result = remote_detect(image, url="http://gpu:5000")

**v2:** Built-in ``pyzm.serve`` + gateway mode:

.. code-block:: bash

   # Start the server
   python -m pyzm.serve --models yolov4 --port 5000

.. code-block:: python

   # Client code -- identical Detector API
   detector = Detector(models=["yolov4"], gateway="http://gpu:5000")
   result = detector.detect("/path/to/image.jpg")

See :doc:`/guide/serve` for details.


What's preserved
-----------------

These legacy modules are still importable for backward compatibility:

- ``pyzm.ZMLog`` -- legacy logging (wraps ``setup_logging`` internally)
- ``pyzm.ZMMemory`` -- shared-memory access
- ``pyzm.ZMEventNotification`` -- websocket notifications
- ``pyzm.api`` -- the old ``ZMApi`` class


What's new in v2
-----------------

- ``ZMClient.tag_event(event_id, labels)`` -- tag events with detected objects (ZM >= 1.37.44)
- ``ZMClient.event_path(event_id)`` -- get filesystem path for an event
- ``DetectionResult.annotate()`` -- draw bounding boxes on the detection image
- ``DetectionResult.summary`` -- human-readable one-liner (``"person:97% car:85%"``)
- ``pyzm.serve`` -- built-in remote ML detection server
- Pydantic v2 config models with validation
