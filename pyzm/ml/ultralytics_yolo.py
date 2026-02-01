import os
import numpy as np
from pyzm.helpers.Base import Base
import portalocker
from pyzm.helpers.utils import Timer
import pyzm.helpers.globals as g


class UltralyticsDetector(Base):

    def __init__(self, options={}):
        self.model = None
        self.classes = None
        self.options = options
        self.is_locked = False
        self.processor = self.options.get('object_processor') or 'cpu'
        self.lock_maximum = int(options.get(self.processor + '_max_processes') or 1)
        self.lock_timeout = int(options.get(self.processor + '_max_lock_wait') or 120)

        self.lock_name = 'pyzm_uid{}_{}_lock'.format(os.getuid(), self.processor)

        self.disable_locks = options.get('disable_locks', 'no')
        if self.disable_locks == 'no':
            g.logger.Debug(2, 'portalock: max:{}, name:{}, timeout:{}'.format(
                self.lock_maximum, self.lock_name, self.lock_timeout))
            self.lock = portalocker.BoundedSemaphore(
                maximum=self.lock_maximum, name=self.lock_name,
                timeout=self.lock_timeout)

        self.imgsz = int(self.options.get('model_width', 640))

    def acquire_lock(self):
        if self.disable_locks == 'yes':
            return
        if self.is_locked:
            g.logger.Debug(2, '{} portalock already acquired'.format(self.lock_name))
            return
        try:
            g.logger.Debug(2, 'Waiting for {} portalock...'.format(self.lock_name))
            self.lock.acquire()
            g.logger.Debug(2, 'Got {} portalock'.format(self.lock_name))
            self.is_locked = True
        except portalocker.AlreadyLocked:
            g.logger.Error('Timeout waiting for {} portalock for {} seconds'.format(
                self.lock_name, self.lock_timeout))
            raise ValueError(
                'Timeout waiting for {} portalock for {} seconds'.format(
                    self.lock_name, self.lock_timeout))

    def release_lock(self):
        if self.disable_locks == 'yes':
            return
        if not self.is_locked:
            g.logger.Debug(2, '{} portalock already released'.format(self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        g.logger.Debug(2, 'Released {} portalock'.format(self.lock_name))

    def get_options(self):
        return self.options

    def populate_class_labels(self):
        class_file = self.options.get('object_labels')
        if class_file:
            f = open(class_file, 'r')
            self.classes = [line.strip() for line in f.readlines()]
            f.close()
        else:
            # Use built-in names from the ultralytics model
            self.classes = list(self.model.names.values())

    def get_classes(self):
        return self.classes

    def load_model(self):
        from ultralytics import YOLO

        name = self.options.get('name') or 'Ultralytics'
        weights_path = self.options.get('object_weights')

        if not os.path.isfile(weights_path):
            raise ValueError(
                'Ultralytics weights file not found: {}. '
                'Please download it manually.'.format(weights_path))

        g.logger.Debug(1, '|--------- Loading "{}" model from disk -------------|'.format(name))
        t = Timer()
        self.model = YOLO(weights_path)
        diff_time = t.stop_and_get_ms()

        if self.processor == 'gpu':
            self.device = 0
            g.logger.Debug(1, 'Using GPU (CUDA device 0) for Ultralytics detection')
        else:
            self.device = 'cpu'
            g.logger.Debug(1, 'Using CPU for Ultralytics detection')

        g.logger.Debug(
            1, 'perf: processor:{} Ultralytics initialization (loading {} model from disk) took: {}'
            .format(self.processor, weights_path, diff_time))

        self.populate_class_labels()

    def detect(self, image=None):
        Height, Width = image.shape[:2]
        g.logger.Debug(2, 'detect extracted image dimensions as: {}wx{}h'.format(Width, Height))

        if self.options.get('auto_lock', True):
            self.acquire_lock()

        try:
            if not self.model:
                self.load_model()

            conf_threshold = float(self.options.get('object_min_confidence', 0.3))
            nms_threshold = 0.4

            g.logger.Debug(
                1, '|---------- Ultralytics (input image: {}w*{}h, imgsz: {}) ----------|'
                .format(Width, Height, self.imgsz))

            t = Timer()
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=nms_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            if self.options.get('auto_lock', True):
                self.release_lock()
        except:
            if self.options.get('auto_lock', True):
                self.release_lock()
            raise

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1, 'perf: processor:{} Ultralytics detection took: {}'.format(
                self.processor, diff_time))

        bbox = []
        label = []
        conf = []

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    bbox.append([int(round(x1)), int(round(y1)),
                                 int(round(x2)), int(round(y2))])
                    label.append(str(self.classes[cls_ids[i]]))
                    conf.append(float(confs[i]))

        weights_file = os.path.basename(self.options.get('object_weights', 'ultralytics'))
        model_tag = 'ultralytics:{}'.format(weights_file)
        return bbox, label, conf, [model_tag] * len(label)
