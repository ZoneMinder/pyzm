Model Fine-Tuning
=================

pyzm includes a web-based UI for fine-tuning YOLO object detection models on
your own data. This lets ZoneMinder users teach the model to recognize custom
objects (e.g. specific vehicles, pets, packages) or improve detection accuracy
for objects the base model struggles with.

The UI is built on Streamlit and runs as a local web app. It guides you
through the full workflow: importing images, reviewing/correcting detections,
and training a fine-tuned model that can be exported as ONNX for use with
ZoneMinder's event notification system.

Installation
------------

.. code-block:: bash

   pip install --break-system-packages "pyzm[train]"

This installs Ultralytics (YOLO), Streamlit, and the canvas/image components
used by the UI.

Launching the UI
----------------

.. code-block:: bash

   python -m pyzm.train

Or directly via Streamlit:

.. code-block:: bash

   streamlit run pyzm/train/app.py -- --base-path /var/lib/zmeventnotification/models

Options:

``--base-path``
   Path to your ZoneMinder models directory.
   Default: ``/var/lib/zmeventnotification/models``

``--workspace-dir``
   Override the project storage directory.
   Default: ``~/.pyzm/training``

``--processor``
   ``gpu`` or ``cpu`` for auto-detection. Default: ``gpu``

``--host``
   Bind address. Default: ``0.0.0.0``

``--port``
   Port. Default: ``8501``

Workflow
--------

The training UI has three phases:

1. Select Images
^^^^^^^^^^^^^^^^

Import training images from one of three sources:

- **Pre-Annotated YOLO Dataset** -- import an existing dataset in YOLO format
  (e.g. from Roboflow). The UI reads the ``data.yaml`` and imports images with
  their annotations already attached.

- **Raw Images** -- upload unannotated images (JPG, PNG, etc.). The base model
  runs auto-detection on each image, giving you a starting point to correct.

- **ZoneMinder Events** -- connect to your ZM instance, browse monitors and
  events, and import frames directly. This is the most natural workflow for
  fixing detection problems: find an event where the model failed, import the
  frame, and correct the labels.

When the UI detects that certain classes need more training images (based on
your review corrections), it shows a banner at the top of this phase listing
which classes need attention and their current progress.

2. Review Detections
^^^^^^^^^^^^^^^^^^^^

For each imported image, review the auto-detected bounding boxes:

- **Approve** -- the detection is correct
- **Delete** -- remove a false positive
- **Rename** -- change the label (e.g. "car" should be "truck")
- **Reshape** -- drag/resize a box that's too large or misaligned
- **Add** -- draw new boxes for objects the model missed

The sidebar shows an image navigator and per-class coverage (how many images
contain each class vs. the minimum needed for training).

You can expand the canvas for more precise drawing, and clear drawn boxes if
you make a mistake.

3. Train & Export
^^^^^^^^^^^^^^^^^

Once all images are reviewed and classes have enough data:

- Configure training parameters (epochs, batch size, image size)
- The UI auto-detects GPU/CPU and suggests an appropriate batch size
- Click **Start Training** to begin fine-tuning
- A live progress bar shows epoch, loss curves, and mAP metrics
- Training logs are displayed in real time

After training completes, the UI shows:

- **Results summary** -- mAP50, mAP50-95, model size, training time
- **Per-class metrics** -- precision, recall, and AP for each class
- **Training analysis** -- interpretive guidance on the quality of your model,
  training curves, confusion matrix, F1/PR curves, and validation samples
- **Export** -- export the fine-tuned model as ONNX with a ready-to-use
  snippet for your ``objectconfig.yml``

Projects
--------

The training UI supports multiple projects. Each project stores its images,
annotations, verification state, and training runs independently under
``~/.pyzm/training/<project-name>/``.

When you launch the UI, you can create a new project or resume an existing
one. The **Switch Project** and **Reset Project** buttons in the sidebar let
you manage projects.

Using the fine-tuned model
--------------------------

After exporting, copy the ONNX file to your models directory and update your
``objectconfig.yml``:

.. code-block:: yaml

   models:
     - name: my_finetune
       type: object
       framework: opencv
       weights: /var/lib/zmeventnotification/models/custom_finetune/yolo11s_finetune.onnx
       min_confidence: 0.3
       pattern: "(dog|cat|package)"

The fine-tuned model can be used alongside your existing models. pyzm's
detection pipeline will run all configured models and merge results according
to your ``match_strategy``.

Tips
----

- **Start small** -- 10-20 images per class is enough for a first pass. You
  can always add more and retrain.
- **Use ZM events** -- importing frames from events where detection failed is
  the most effective way to improve accuracy for your specific camera angles
  and lighting.
- **Check the confusion matrix** -- it shows which classes the model confuses
  with each other, helping you decide where to add more data.
- **Watch for overfitting** -- if the best model epoch is early in training
  (e.g. epoch 15 of 50), try fewer epochs or more training data.
- **Export and test** -- use the built-in test image feature to verify the
  fine-tuned model before deploying it.
