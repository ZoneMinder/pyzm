pyzm package
============

Legacy API
-----------

The original ZoneMinder API wrapper. For new code, prefer :class:`~pyzm.client.ZMClient`.

.. automodule:: pyzm.api
    :members:
    :special-members: __init__
    :undoc-members:
    :show-inheritance:

Legacy helper classes
~~~~~~~~~~~~~~~~~~~~~~

Wrapper objects returned by the legacy ``pyzm.api.ZMApi`` class.

.. automodule:: pyzm.helpers.Monitors
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pyzm.helpers.Monitor
    :members:
    :undoc-members:
    :show-inheritance:


.. automodule:: pyzm.helpers.Events
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pyzm.helpers.Event
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pyzm.helpers.States
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pyzm.helpers.State
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: pyzm.helpers.Base
    :members:
    :undoc-members:
    :show-inheritance:

Logging
--------

Logging utilities for pyzm. ``setup_zm_logging()`` provides ZM-native logging
(file, database, syslog) matching Perl's Logger.pm format.

.. automodule:: pyzm.log
    :members: setup_zm_logging, ZMLogAdapter
    :undoc-members:

Event Notification
---------------------

WebSocket client for receiving real-time event notifications from the
ZoneMinder Event Server (``zmeventnotification.pl``).

.. automodule:: pyzm.ZMEventNotification
    :members:
    :special-members: __init__
    :undoc-members:

Memory
-------

Direct access to ZoneMinder's shared memory segments for low-latency
monitor state and trigger data.

.. automodule:: pyzm.ZMMemory
    :members:
    :special-members: __init__
    :undoc-members:


Machine Learning
------------------

The ML detection pipeline. ``Detector`` is the main entry point --
it manages backends, model sequencing, and result filtering.

.. automodule:: pyzm.ml.detector
    :members:
    :special-members: __init__
    :undoc-members:

.. automodule:: pyzm.ml.pipeline
    :members:
    :special-members: __init__
    :undoc-members:

.. automodule:: pyzm.ml.filters
    :members:
    :undoc-members:

Configuration Models
---------------------

Pydantic v2 models for all pyzm configuration: ZM client settings,
detector/model parameters, and stream extraction options.

.. automodule:: pyzm.models.config
    :members:
    :undoc-members:

.. automodule:: pyzm.models.detection
    :members:
    :undoc-members:

.. automodule:: pyzm.models.zm
    :members:
    :undoc-members:

Remote ML Detection Server
----------------------------

A FastAPI-based server that loads models once and serves detection requests
over HTTP. See the :doc:`serve guide </guide/serve>` for usage.

.. automodule:: pyzm.serve.app
    :members:
    :undoc-members:

.. automodule:: pyzm.serve.auth
    :members:
    :undoc-members:

ZoneMinder Client
------------------

The v2 typed client for the ZoneMinder REST API. Returns dataclass models
(``Monitor``, ``Event``, ``Zone``) instead of raw dicts.

.. automodule:: pyzm.client
    :members:
    :special-members: __init__
    :undoc-members:


