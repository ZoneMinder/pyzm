import numpy as np
import cv2
import re
from pyzm.helpers.Base import Base
import portalocker
import os
from pyzm.helpers.utils import Timer
import pyzm.helpers.globals as g


# cv2 version check for unconnected layers fix
def cv2_version() -> tuple:
    # Returns (major, minor, patch) as ints for proper numeric comparison
    x = cv2.__version__.split(".")
    # Strip non-numeric suffixes like "-dev"
    parts = [re.sub(r'[^0-9]', '', p) or '0' for p in x]
    maj = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (maj, minor, patch)


class YoloBase(Base):
    """Shared base class for YOLO backends (Darknet and ONNX).

    Subclasses must implement:
      - load_model()
      - _forward_and_parse(blob, Width, Height, conf_threshold)
            -> (class_ids, confidences, boxes)
    """

    # opencv DNN code credit: https://github.com/arunponnusamy/cvlib

    def __init__(self, options={}, default_dim=416):
        self.net = None
        self.classes = None
        self.options = options
        self.is_locked = False
        self.processor = self.options.get('object_processor') or 'cpu'
        self.lock_maximum = int(options.get(self.processor + '_max_processes') or 1)
        self.lock_timeout = int(options.get(self.processor + '_max_lock_wait') or 120)
        self.name = self.options.get('name') or 'Yolo'

        self.lock_name = 'pyzm_uid{}_{}_lock'.format(os.getuid(), self.processor)

        self.disable_locks = options.get('disable_locks', 'no')
        if self.disable_locks == 'no':
            g.logger.Debug(2, '{}: portalock: max:{}, name:{}, timeout:{}'.format(self.name, self.lock_maximum, self.lock_name,
                                                                              self.lock_timeout))
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,
                                                     timeout=self.lock_timeout)

        self.model_height = self.options.get('model_height', default_dim)
        self.model_width = self.options.get('model_width', default_dim)

    def acquire_lock(self):
        if self.disable_locks == 'yes':
            return
        if self.is_locked:
            g.logger.Debug(2, '{}: {} portalock already acquired'.format(self.name, self.lock_name))
            return
        try:
            g.logger.Debug(2, '{}: Waiting for {} portalock...'.format(self.name, self.lock_name))
            self.lock.acquire()
            g.logger.Debug(2, '{}: Got {} portalock'.format(self.name, self.lock_name))
            self.is_locked = True

        except portalocker.AlreadyLocked:
            g.logger.Error('{}: Timeout waiting for {} portalock for {} seconds'.format(self.name, self.lock_name, self.lock_timeout))
            raise ValueError(
                '{}: Timeout waiting for {} portalock for {} seconds'.format(self.name, self.lock_name, self.lock_timeout))

    def release_lock(self):
        if self.disable_locks == 'yes':
            return
        if not self.is_locked:
            g.logger.Debug(2, '{}: {} portalock already released'.format(self.name, self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        g.logger.Debug(2, '{}: Released {} portalock'.format(self.name, self.lock_name))

    def get_options(self):
        return self.options

    def get_classes(self):
        return self.classes

    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        f = open(class_file_abs_path, 'r')
        self.classes = [line.strip() for line in f.readlines()]
        f.close()

    def _setup_gpu(self, cv2_ver):
        """Configure CUDA backend if processor is 'gpu' and OpenCV supports it."""
        if self.processor == 'gpu':
            if cv2_ver < (4, 2, 0):
                g.logger.Error('{}: Not setting CUDA backend for OpenCV DNN'.format(self.name))
                g.logger.Error(
                    '{}: OpenCV version {} does not support CUDA for DNNs. A minimum of 4.2 is required. See https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/ on how to compile and install openCV 4.2'
                    .format(self.name, cv2.__version__))
                self.processor = 'cpu'
        else:
            g.logger.Debug(1, '{}: Using CPU for detection'.format(self.name))

        if self.processor == 'gpu':
            g.logger.Debug(2, '{}: Setting CUDA backend for OpenCV'.format(self.name))
            g.logger.Debug(3,
                           '{}: If you did not set your CUDA_ARCH_BIN correctly during OpenCV compilation, you will get errors during detection related to invalid device/make_policy'.format(self.name))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def load_model(self):
        raise NotImplementedError

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        raise NotImplementedError

    def _create_blob(self, image):
        scale = 0.00392  # 1/255, normalize to [0, 1]
        return cv2.dnn.blobFromImage(image,
                                     scale, (self.model_width, self.model_height), (0, 0, 0),
                                     True,
                                     crop=False)

    def detect(self, image=None):
        Height, Width = image.shape[:2]
        g.logger.Debug(2, '{}: detect extracted image dimensions as: {}wx{}h'.format(self.name, Width, Height))

        if self.options.get('auto_lock', True):
            self.acquire_lock()

        try:
            if not self.net:
                self.load_model()

            g.logger.Debug(
                1, '|---------- {} (input image: {}w*{}h, model resize dimensions: {}w*{}h) ----------|'
                .format(self.name, Width, Height, self.model_width, self.model_height))

            t = Timer()
            blob = self._create_blob(image)

            nms_threshold = 0.4
            conf_threshold = 0.2

            # first nms filter out with a yolo confidence of 0.2 (or less)
            if float(self.options.get('object_min_confidence')) < conf_threshold:
                conf_threshold = float(self.options.get('object_min_confidence'))

            class_ids, confidences, boxes = self._forward_and_parse(blob, Width, Height, conf_threshold)

            if self.options.get('auto_lock', True):
                self.release_lock()
        except:
            if self.options.get('auto_lock', True):
                self.release_lock()
            raise

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1, 'perf: processor:{} {} detection took: {}'.format(self.processor, self.name, diff_time))

        t = Timer()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            2, 'perf: processor:{} {} NMS filtering took: {}'.format(self.processor, self.name, diff_time))
        # NMSBoxes returns flat indices in OpenCV >= 4.5.4, nested [[i]] before that
        indices = np.array(indices).flatten()

        bbox = []
        label = []
        conf = []

        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            bbox.append([
                int(round(x)),
                int(round(y)),
                int(round(x + w)),
                int(round(y + h))
            ])
            label.append(str(self.classes[class_ids[i]]))
            conf.append(confidences[i])

        return bbox, label, conf, ['yolo'] * len(label)


def Yolo(options={}):
    """Factory function: returns YoloDarknet or YoloOnnx based on weights file extension."""
    weights_path = options.get('object_weights', '')
    if weights_path.lower().endswith('.onnx'):
        from pyzm.ml.yolo_onnx import YoloOnnx
        return YoloOnnx(options)
    else:
        from pyzm.ml.yolo_darknet import YoloDarknet
        return YoloDarknet(options)
