from os import getuid
from pathlib import Path
from typing import Optional

from PIL import Image

import cv2
# Pycharm hack for intellisense
# from cv2 import cv2
import portalocker
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

from pyzm.helpers.pyzm_utils import Timer, str2bool
from pyzm.interface import GlobalConfig
from pyzm.ml.face import Face

g: GlobalConfig
name = 'TPU_Face'


# Class to handle face recognition
class FaceTpu(Face):
    def __init__(self, options=None, *args, **kwargs):
        global g
        g = GlobalConfig()
        g.logger.debug(4, f'TPU Face init params: {options}')
        self.knn = None
        if options is None:
            options = {}
        self.options = options
        self.sequence_name: str = ''
        # g.logger.Debug('Initializing face detection')
        self.processor = 'tpu'
        self.lock_maximum = int(options.get(f'{self.processor}_max_processes', 1))
        self.lock_name = f"pyzm_uid{getuid()}_{self.processor.upper()}_lock"
        self.lock_timeout = int(options.get(f'{self.processor}_max_lock_wait', 120))
        self.disable_locks = options.get('disable_locks', 'no')
        if not str2bool(self.disable_locks):
            g.logger.debug(2, f"portalock: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}")
            self.lock = portalocker.BoundedSemaphore(
                maximum=self.lock_maximum,
                name=self.lock_name,
                timeout=self.lock_timeout
            )
        self.is_locked = False
        self.model = None

    def get_options(self):
        return self.options

    def acquire_lock(self):
        if str2bool(self.disable_locks):
            return
        if self.is_locked:
            g.logger.debug(2, f"portalock: already acquired -> '{self.lock_name}'")
            return
        try:
            # g.logger.Debug (2,f'YOLO: Waiting for portalock: {self.lock_name} ...')
            self.lock.acquire()
            g.logger.debug(2, f"portalock: acquired -> '{self.lock_name}'")
            self.is_locked = True

        except portalocker.AlreadyLocked:
            g.logger.error(f"portalock: Timeout waiting for -> '{self.lock_timeout}' sec: {self.lock_name}")
            raise ValueError(f"portalock: Timeout waiting for {self.lock_timeout} sec: {self.lock_name}")

    def release_lock(self):
        if str2bool(self.disable_locks):
            return
        if not self.is_locked:
            # g.logger.Debug(2, f"portalock: already released: {self.lock_name}")
            return
        self.lock.release()
        self.is_locked = False
        g.logger.debug(2, f"portalock: released -> '{self.lock_name}'")

    def get_classes(self):
        if self.knn:
            return self.knn.classes_
        else:
            return []

    def _rescale_rects(self, a):
        rects = []
        for (left, top, right, bottom) in a:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            rects.append([left, top, right, bottom])
        return rects

    def load_model(self):
        global name
        name = self.options.get('name') or self.get_model_name()
        self.sequence_name = name

        t = Timer()
        self.model = make_interpreter(self.options.get('face_weights'))
        self.model.allocate_tensors()
        diff_time = t.stop_and_get_ms()
        g.logger.debug(
            f"perf:coral:face: '{name}' loading '{Path(self.options.get('face_weights')).name}' took: {diff_time}")

    def get_model_name(self) -> str:
        return 'Face-TPU'

    def get_sequence_name(self) -> str:
        return self.sequence_name

    def detect(self, input_image):
        Height, Width = input_image.shape[:2]
        img = input_image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.options.get('auto_lock', True):
            self.acquire_lock()

        t = Timer()
        try:
            if not self.model:
                self.load_model()
            g.logger.debug(
                f"|***  Face TPU (input image: {Width}*{Height}) ***|")
            _, scale = common.set_resized_input(
                self.model, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
            self.model.invoke()
            objs = detect.get_objects(
                self.model,
                float(self.options.get('face_min_confidence', 0.1)),
                scale
            )
            # outs = self.model.detect_with_image(img, threshold=int(self.options.get('object_min_confidence')),
            #        keep_aspect_ratio=True, relative_coord=False)
            if self.options.get('auto_lock', True):
                self.release_lock()
        except Exception as all_ex:
            if self.options.get('auto_lock', True):
                self.release_lock()
            raise

        diff_time = t.stop_and_get_ms()
        g.logger.debug(
            f"perf:coral:face: '{name}' detection took: {diff_time}")

        bbox = []
        labels = []
        conf = []

        for obj in objs:
            # box = obj.bbox.flatten().astype("int")
            bbox.append([
                int(round(obj.bbox.xmin)),
                int(round(obj.bbox.ymin)),
                int(round(obj.bbox.xmax)),
                int(round(obj.bbox.ymax))
            ])

            labels.append(self.options.get('unknown_face_name', 'face'))
            conf.append(float(obj.score))
        g.logger.debug(3, f"coral:face: returning -> {labels} {bbox} {conf}")
        return bbox, labels, conf, ['face_tpu'] * len(labels)
