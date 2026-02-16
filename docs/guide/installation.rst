Installation
============

Requirements
------------

- Python 3.10 or newer
- Pydantic >= 2.0
- OpenCV 4.13+ (``cv2``) -- required for ML detection features (ONNX YOLO models need 4.13+ for full operator support)
- A running ZoneMinder instance (for API features)

.. note::

   On newer Linux distributions (Debian 12+, Ubuntu 23.04+, Fedora 38+),
   the system Python is marked as *externally managed* and ``pip install``
   will fail unless you add ``--break-system-packages``. All ``pip`` commands
   below include this flag. If you are using a virtual environment, you can
   omit it.

Installing from PyPI
--------------------

**Core library** (API client, ML detection, logging):

.. code-block:: bash

   pip install --break-system-packages pyzm

**With the remote ML detection server** (adds FastAPI, Uvicorn):

.. code-block:: bash

   pip install --break-system-packages "pyzm[serve]"

**With the model fine-tuning UI** (adds Ultralytics, Streamlit):

.. code-block:: bash

   pip install --break-system-packages "pyzm[train]"

**Everything** (core + serve + train):

.. code-block:: bash

   pip install --break-system-packages "pyzm[serve,train]"

What each extra installs
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Extra
     - Additional packages
   * - *(core)*
     - requests, pydantic, numpy, Pillow, PyYAML, onnx, and others
   * - ``[serve]``
     - fastapi, uvicorn, python-multipart, PyJWT
   * - ``[train]``
     - ultralytics, streamlit, streamlit-drawable-canvas, st-clickable-images

.. note::

   The ``[train]`` extra pulls in ``ultralytics``, which installs its own
   ``opencv-python`` from PyPI. If you need GPU-accelerated or Apple Silicon
   OpenCV (e.g. built from source with CUDA or Metal support), install the
   training extras first, then remove the pip OpenCV and install your custom
   build:

   .. code-block:: bash

      pip install --break-system-packages "pyzm[train]"
      pip uninstall opencv-python opencv-python-headless
      # Now build/install OpenCV from source — see below

Building OpenCV from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pip ``opencv-python`` package is CPU-only. For GPU-accelerated inference
you need to build OpenCV from source:

- `Apple Silicon (macOS) <https://gist.github.com/pliablepixels/d0605aab085592e1d3b6bb9033bdc835>`_
- `Ubuntu 24.04 with CUDA <https://gist.github.com/pliablepixels/73d61e28060c8d418f9fcfb1e912e425>`_

Installing from GitHub
-----------------------

To install the latest development version directly from GitHub:

.. code-block:: bash

   pip install --break-system-packages "git+https://github.com/pliablepixels/pyzm.git"

With extras:

.. code-block:: bash

   pip install --break-system-packages "pyzm[train] @ git+https://github.com/pliablepixels/pyzm.git"

Optional dependencies
---------------------

Some ML backends have additional requirements that are **not** installed
automatically:

- (DEPRECATED) **Coral EdgeTPU** -- ``pycoral`` and the Edge TPU runtime
- **Face recognition (dlib)** -- ``dlib``, ``face_recognition``
- **ALPR** -- a running OpenALPR or Plate Recognizer service
- **AWS Rekognition** -- ``boto3`` with configured AWS credentials

Install these manually as needed for your detection pipeline.

.. note::

   Google has discontinued the Coral EdgeTPU product line and the
   ``pycoral`` library is no longer maintained. Installing it on modern
   Python (3.10+) and recent Linux distributions requires manual
   workarounds. See `pycoral#149
   <https://github.com/google-coral/pycoral/issues/149>`_ for
   community discussion and installation tips.

Verifying the installation
--------------------------

.. code-block:: bash

   python -c "import pyzm; print(pyzm.__version__)"

To verify the training UI is available:

.. code-block:: bash

   python -c "from pyzm.train import check_dependencies; check_dependencies(); print('OK')"
