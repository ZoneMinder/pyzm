Remote ML Detection Server
===========================

``pyzm.serve`` is a built-in FastAPI server that loads ML models once
and serves detection requests over HTTP. This lets you offload GPU-heavy
inference to a dedicated machine while your ZoneMinder box sends
lightweight image uploads.

.. code-block:: text

   Image mode (default)                   GPU box
   +-----------------+     HTTP/JPEG     +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |   YoloV4 (GPU)   |
   |   gateway=...)  | <---------------- |   Coral TPU      |
   +-----------------+  DetectionResult  +------------------+

   URL mode (gateway_mode="url")          GPU box
   +-----------------+   frame URLs      +------------------+
   | zm_detect.py    | ----------------> | pyzm.serve       |
   | Detector(       |                   |  fetch from ZM   |
   |   gateway=...,  | <---------------- |  detect & return |
   |   gateway_mode= |  DetectionResult  +------------------+
   |     "url")      |                          |
   +-----------------+                   +------v-----------+
                                         | ZoneMinder API   |
                                         +------------------+

Two detection modes are available:

- **Image mode** (default) -- the client fetches frames from ZM, JPEG-encodes
  them, and uploads each one to the ``/detect`` endpoint.
- **URL mode** -- the client sends frame URLs to the ``/detect_urls`` endpoint
  and the *server* fetches images directly from ZoneMinder. This avoids
  transferring every frame through the client when the server has faster or
  direct network access to ZM.


Server setup
-------------

Install with the ``serve`` extra:

.. code-block:: bash

   pip install pyzm[serve]

Start the server:

.. code-block:: bash

   python -m pyzm.serve --models yolov4 --port 5000

With GPU and authentication:

.. code-block:: bash

   python -m pyzm.serve \
       --models yolov4 \
       --processor gpu \
       --port 5000 \
       --auth --auth-user admin --auth-password secret

CLI options
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--host``
     - ``0.0.0.0``
     - Bind address
   * - ``--port``
     - ``5000``
     - Bind port
   * - ``--models``
     - ``yolov4``
     - Model names (space-separated)
   * - ``--base-path``
     - ``/var/lib/zmeventnotification/models``
     - Directory containing model subdirectories
   * - ``--processor``
     - ``cpu``
     - ``cpu``, ``gpu``, or ``tpu``
   * - ``--auth``
     - off
     - Enable JWT authentication
   * - ``--auth-user``
     - ``admin``
     - Username (when auth enabled)
   * - ``--auth-password``
     - (empty)
     - Password (when auth enabled)
   * - ``--token-expiry``
     - ``3600``
     - JWT token expiry in seconds


Client usage
-------------

Image mode (default)
~~~~~~~~~~~~~~~~~~~~~

Using the ``Detector`` API:

.. code-block:: python

   from pyzm import Detector

   detector = Detector(models=["yolov4"], gateway="http://gpu-box:5000")
   result = detector.detect("/path/to/image.jpg")

   print(result.summary)

The ``Detector`` JPEG-encodes the image and sends it to the server.
The response is deserialized into a standard ``DetectionResult``.

With authentication:

.. code-block:: python

   detector = Detector(
       models=["yolov4"],
       gateway="http://gpu-box:5000",
       gateway_username="admin",
       gateway_password="secret",
   )

URL mode
~~~~~~~~~

When ``gateway_mode="url"`` is set, ``detect_event()`` sends frame URLs
to the server instead of uploading JPEG data. The server fetches images
directly from ZoneMinder using the provided auth token:

.. code-block:: python

   detector = Detector(
       models=["yolov4"],
       gateway="http://gpu-box:5000",
       gateway_mode="url",
   )

   # detect_event() builds ZM frame URLs and POSTs them to /detect_urls
   result = detector.detect_event(zm_client, event_id=12345,
                                   stream_config=stream_cfg)

URL mode only applies to ``detect_event()`` calls.  Single-image
``detect()`` calls always use image mode regardless of this setting.

This is useful when:

- The server has faster or more direct network access to ZoneMinder
- You want to avoid transferring every frame through the client
- The client machine has limited upload bandwidth

Using ``from_dict()``
~~~~~~~~~~~~~~~~~~~~~~

The ``ml_gateway`` key in the ``general`` section of ``ml_options``
automatically enables remote mode.  Set ``ml_gateway_mode`` to ``"url"``
to use URL mode:

.. code-block:: python

   ml_options = {
       "general": {
           "model_sequence": "object",
           "ml_gateway": "http://gpu-box:5000",
           "ml_gateway_mode": "url",        # "image" (default) or "url"
           # "ml_user": "admin",
           # "ml_password": "secret",
       },
       "object": {
           "general": {"pattern": ".*"},
           "sequence": [...],
       },
   }

   detector = Detector.from_dict(ml_options)
   result = detector.detect(image)


Authentication
---------------

When the server is started with ``--auth``, clients must first obtain a
JWT token via ``/login``, then pass it as a Bearer token on subsequent
requests. The ``Detector`` gateway mode handles this automatically.

Manual flow:

.. code-block:: bash

   # Login
   TOKEN=$(curl -s -X POST http://gpu-box:5000/login \
       -H 'Content-Type: application/json' \
       -d '{"username":"admin","password":"secret"}' \
       | jq -r .access_token)

   # Detect
   curl -X POST http://gpu-box:5000/detect \
       -H "Authorization: Bearer $TOKEN" \
       -F file=@/path/to/image.jpg

Tokens expire after ``--token-expiry`` seconds (default 3600).


API reference
--------------

``GET /health``
~~~~~~~~~~~~~~~~

Health check. Returns:

.. code-block:: json

   {"status": "ok", "models_loaded": true}

``POST /detect``
~~~~~~~~~~~~~~~~~

Run detection on an uploaded image (image mode).

- **Content-Type:** ``multipart/form-data``
- **Parameters:**
  - ``file`` (required) -- JPEG/PNG image
  - ``zones`` (optional) -- JSON string of zone list
- **Auth:** Bearer token (when auth enabled)
- **Returns:** ``DetectionResult`` as JSON (image field excluded)

``POST /detect_urls``
~~~~~~~~~~~~~~~~~~~~~~

Run detection on images fetched from URLs (URL mode).

- **Content-Type:** ``application/json``
- **Body:**

  .. code-block:: json

     {
       "urls": [
         {"frame_id": "snapshot", "url": "https://zm.example.com/zm/index.php?view=image&eid=123&fid=snapshot"},
         {"frame_id": "1", "url": "https://zm.example.com/zm/index.php?view=image&eid=123&fid=1"}
       ],
       "zm_auth": "token=abc123...",
       "zones": [{"name": "driveway", "value": [[0,0],[100,0],[100,100],[0,100]]}],
       "verify_ssl": false
     }

- **Auth:** Bearer token (when auth enabled)
- **Returns:** ``DetectionResult`` as JSON (best frame selected by ``frame_strategy``)

The server appends ``zm_auth`` to each URL and fetches the image via HTTP GET.
Frame strategy (``first``, ``most``, ``most_unique``, ``most_models``)
is applied server-side to pick the best result.

``POST /login``
~~~~~~~~~~~~~~~~

Obtain a JWT token (only available when ``--auth`` is enabled).

- **Content-Type:** ``application/json``
- **Body:** ``{"username": "...", "password": "..."}``
- **Returns:** ``{"access_token": "...", "expires": 3600}``


objectconfig.yml remote section
---------------------------------

In a ZoneMinder event notification setup, configure the remote gateway
in ``objectconfig.yml``:

.. code-block:: yaml

   ml_sequence:
     general:
       model_sequence: "object"
       ml_gateway: "http://gpu-box:5000"
       ml_gateway_mode: "url"            # "image" (default) or "url"
       ml_gateway_username: "admin"
       ml_gateway_password: "secret"
       ml_fallback_local: yes

     object:
       general:
         pattern: "(person|car)"
       sequence:
         - object_framework: opencv
           object_weights: /path/to/yolov4.weights
           object_config: /path/to/yolov4.cfg
           object_labels: /path/to/coco.names

When ``ml_gateway`` is set, detection requests are sent to the remote
server. If ``ml_fallback_local`` is ``yes`` and the remote server is
unreachable, detection falls back to local inference using the
configured model sequence.

Set ``ml_gateway_mode`` to ``url`` when the GPU box has direct access
to ZoneMinder. In this mode, the client sends frame URLs instead of
uploading JPEG data, and the server fetches images directly from ZM.
