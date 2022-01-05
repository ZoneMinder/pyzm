from os import getuid
from typing import Optional
from PIL import Image

import cv2
# Pycharm hack for intellisense
# from cv2 import cv2
import numpy as np

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pathlib import Path

from pyzm.ml.object import Object
from pyzm.helpers.pyzm_utils import Timer
from pyzm.interface import GlobalConfig

g: GlobalConfig
lp: str


class TPUException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        super().__init__(self.message)

    def __str__(self):
        if self.message:
            return self.message
        else:
            return "TPUException has been raised"


class Tpu(Object):
    def __init__(self, *args, **kwargs):
        global g, lp
        self.lp = lp = 'coral:'
        g = GlobalConfig()
        options = kwargs['options']
        kwargs['globs'] = g

        super().__init__(*args, **kwargs)

        if options is None:
            g.logger.error(
                f"{lp} cannot initialize TPU object detection model -> no 'sequence' sent with weights, conf, "
                f"labels, etc.")
            raise TPUException(f"{lp} -> NO OPTIONS")
        g.logger.debug(f"{lp} initializing edge TPU with params: {options}")
        self.sequence_name: str = ''

        self.classes = {}
        self.options = options
        self.processor = 'tpu'
        self.lock_maximum = int(options.get(f"{self.processor}_max_processes", 1))
        self.lock_name = f"pyzm_uid{getuid()}_{self.processor.upper()}_lock"
        self.lock_timeout = int(options.get(f"{self.processor}_max_lock_wait", 120))
        self.disable_locks = options.get('disable_locks', 'no')
        self.create_lock()
        self.is_locked = False
        self.model = None
        self.model_height = self.options.get('model_height', 312)
        self.model_width = self.options.get('model_width', 312)
        self.populate_class_labels()

    def get_model_name(self) -> str:
        return 'coral'

    def get_options(self, key=None):
        if not key:
            return self.options
        else:
            return self.options.get(key)

    def get_sequence_name(self) -> str:
        return self.sequence_name

    def populate_class_labels(self):
        label_file = self.options.get('object_labels')
        if label_file and Path(label_file).is_file():
            fp = None
            try:
                fp = open(label_file)
            except Exception as exc:
                g.logger.error(f"{lp} error while trying to open the 'object labels' file '{label_file}' -> \n{exc}")
            else:
                for row in fp:
                    # unpack the row and update the labels dictionary
                    (classID, label) = row.strip().split(" ", maxsplit=1)
                    self.classes[int(classID)] = label.strip()
            finally:
                if fp:
                    fp.close()
        elif not Path(label_file).is_file():
            g.logger.error(f"{lp} '{Path(label_file).name}' does not exist or is not an actual file")
            raise TPUException(f"Provided label file does not exist or is not a file! Check the config for any spelling mistakes in the entire Path")
        else:
            g.logger.debug(f"{lp} No label file provided for this model")
            raise TPUException(f"Provided label file does not exist or is not a file! Check the config for any spelling mistakes in the entire Path")


    def get_classes(self):
        return self.classes

    def load_model(self):
        # print(f"{self.options = }")
        self.sequence_name = self.options.get('name') or self.get_model_name()
        g.logger.debug(f"{lp} loading model data from sequence '{self.sequence_name}' ")
        # self.model = DetectionEngine(self.options.get('object_weights'))
        # Initialize the TF interpreter
        t = Timer()
        if Path(self.options.get('object_weights')).is_file():
            try:
                self.model = make_interpreter(self.options.get('object_weights'))
            except Exception as ex:
                ex = repr(ex)
                tokens = ex.split(' ')
                for tok in tokens:
                    if tok.startswith('libedgetpu'):
                        g.logger.info(
                            f"{lp} TPU error detected (replace cable with a short high quality one, dont allow "
                            f"TPU/cable to move around). Reset the USB port or reboot!"
                        )
                        raise TPUException("TPU NO COMM")
            else:
                self.model.allocate_tensors()
                diff_time = t.stop_and_get_ms()
                g.logger.debug(
                    f"perf:{lp} initialization -> loading '{Path(self.options.get('object_weights')).name}' "
                    f"took: {diff_time}")
        else:
            g.logger.error(f"{lp} The supplied model file does not exist or is not an actual file. Can't run detection!")

    @staticmethod
    def _nms(objects, threshold):
        # not used - its already part of TPU core libs it seems
        # credit 
        # https://github.com/google-coral/pycoral/blob/master/examples/small_object_detection.py

        """Returns a list of indexes of objects passing the NMS.
        Args:
            objects: result candidates.
            threshold: the threshold of overlapping IoU to merge the boxes.
        Returns:
            A list of indexes containing the objects that pass the NMS.
        """
        if len(objects) == 1:
            return [0]

        boxes = np.array([o.bbox for o in objects])
        xmins = boxes[:, 0]
        ymins = boxes[:, 1]
        xmaxs = boxes[:, 2]
        ymaxs = boxes[:, 3]

        areas = (xmaxs - xmins) * (ymaxs - ymins)
        scores = [o.score for o in objects]
        idxs = np.argsort(scores)

        selected_idxs = []
        while idxs.size != 0:
            selected_idx = idxs[-1]
            selected_idxs.append(selected_idx)

            overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
            overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
            overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
            overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

            w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
            h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

            intersections = w * h
            unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
            ious = intersections / unions

            idxs = np.delete(
                idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

        return selected_idxs

    def detect(self, input_image=None):
        orig_h, orig_w = h, w = input_image.shape[:2]
        downscaled = False
        upsize_xfactor = None
        upsize_yfactor = None
        model_resize = False
        if self.model_height and self.model_width:
            model_resize = True
        elif self.model_height and not self.model_width:
            self.model_width = self.model_height
            model_resize = True
        elif not self.model_height and self.model_width:
            self.model_height = self.model_width
            model_resize = True

        if model_resize:
            downscaled = True
            g.logger.debug(2, f"{lp} model dimensions requested -> "
                              f"{self.model_width}*{self.model_height}")
            input_image = cv2.resize(input_image, (int(self.model_width), int(self.model_height)), interpolation=cv2.INTER_AREA)
            newHeight, newWidth = input_image.shape[:2]
            # getscaling so we can make correct bounding boxes
            upsize_xfactor = w / newWidth
            upsize_yfactor = h / newHeight

        h, w = input_image.shape[:2]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        if self.options.get('auto_lock', True):
            self.acquire_lock()

        try:
            if not self.model:
                self.load_model()

            g.logger.debug(
                f"{lp} '{self.sequence_name}' input image (w*h): {orig_w}*{orig_h} resized by model_width/height "
                f"to {self.model_width}*{self.model_height}"
            )
            t = Timer()
            _, scale = common.set_resized_input(
                self.model, input_image.size, lambda size: input_image.resize(size, Image.ANTIALIAS)
            )
            self.model.invoke()

            # outs = self.model.detect_with_image(img, threshold=int(self.options.get('object_min_confidence')),
            #        keep_aspect_ratio=True, relative_coord=False)
        except Exception as ex:
            raise ex
        else:
            objs = detect.get_objects(self.model, float(self.options.get('object_min_confidence')), scale)
            diff_time = t.stop_and_get_ms()
            g.logger.debug(f"perf:{lp} '{self.sequence_name}' detection took: {diff_time}")

        finally:
            if self.options.get('auto_lock', True):
                self.release_lock()

        bbox, labels, conf = [], [], []

        for obj in objs:
            # box = obj.bbox.flatten().astype("int")
            bbox.append([
                int(round(obj.bbox.xmin)),
                int(round(obj.bbox.ymin)),
                int(round(obj.bbox.xmax)),
                int(round(obj.bbox.ymax))
            ])

            labels.append(self.classes.get(obj.id))
            conf.append(float(obj.score))
        if downscaled and labels:
            # fixme: what if its up scaled?
            g.logger.debug(2,
                           f"{lp} The image was resized before processing by the 'model width/height', scaling "
                           f"bounding boxes in image back up by factors of -> x={upsize_xfactor:.4} "
                           f"y={upsize_yfactor:.4}"
                           )
            bbox = self.downscale(bbox, upsize_xfactor, upsize_yfactor)


        if labels:
            g.logger.debug(f"{lp} returning {labels} -- {bbox} -- {conf}")
        else:
            g.logger.debug(f"{lp} no detections to return!")
        return bbox, labels, conf, ['coral'] * len(labels)  # , ret_val
