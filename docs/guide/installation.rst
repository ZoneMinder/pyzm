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

.. code-block:: bash

   pip install --break-system-packages pyzm

To include the remote ML detection server (FastAPI-based):

.. code-block:: bash

   pip install --break-system-packages pyzm[serve]

Installing from GitHub
-----------------------

To install the latest development version directly from GitHub:

.. code-block:: bash

   pip install --break-system-packages git+https://github.com/pliablepixels/pyzm.git

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
