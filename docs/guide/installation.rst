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

- **Coral EdgeTPU** -- ``pycoral`` and the Edge TPU runtime
- **Face recognition (dlib)** -- ``dlib``, ``face_recognition``
- **ALPR** -- a running OpenALPR or Plate Recognizer service
- **AWS Rekognition** -- ``boto3`` with configured AWS credentials

Install these manually as needed for your detection pipeline.

Verifying the installation
--------------------------

.. code-block:: bash

   python -c "import pyzm; print(pyzm.__version__)"

To verify the training UI is available:

.. code-block:: bash

   python -c "from pyzm.train import check_dependencies; check_dependencies(); print('OK')"
