from copy import deepcopy
from enum import Enum
from os import getuid
from pathlib import Path
from typing import Optional

import cv2
import cv2
# Pycharm hack for intellisense
# from cv2 import cv2
import numpy as np

from pyzm.helpers.pyzm_utils import Timer, str2bool
from pyzm.interface import GlobalConfig
from pyzm.ml.object import Object


lp: Optional[str] = None


class YoloException(Exception):
    def __init__(self, message='My default message', *args, **kwargs):
        print(f"inside custom YoloException init args={args}")
        print(f"kwargs={kwargs}")
        # g.logger.error(message)
        super().__init__(message, *args, **kwargs)



class Yolo(Object):
    # The actual CNN object detection code
    # opencv DNN code credit: https://github.com/arunponnusamy/cvlib
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.lp = lp = 'yolo:'
        globs = kwargs['globs']
        g: GlobalConfig = globs
        self.globs = globs
        self.options: dict = kwargs.get('options', {})
        if self.options is None:
            raise YoloException(f"YOLO no options passed!")
        super().__init__(*args, **kwargs)

        self.original_image: Optional[cv2] = None

        # UnConnectedLayers fix
        self.new_cv_scalar_fix = False
        self.net = None
        self.classes = None
        self.is_locked: bool = False
        self.sequence_name: str = ''
        g.logger.debug(4, f"{lp} initialization params: {self.options}")

        self.processor: str = self.options.get('object_processor', 'cpu')
        self.lock_maximum: int = int(self.options.get(self.processor + '_max_processes', 1))
        self.lock_timeout: int = int(self.options.get(self.processor + '_max_lock_wait', 120))
        self.lock_name = f"pyzm_uid{getuid()}_{self.processor.upper()}_lock"
        self.disable_locks = self.options.get('disable_locks', 'no')
        # method from superclass to reduce duplicate code
        self.create_lock()
        # yolo needs to scale the bounding boxes based on the w/h of the image
        self.model_height = self.options.get('model_height', 416)
        self.model_width = self.options.get('model_width', 416)

    def get_model_name(self) -> str:
        return f"yolo-{self.processor}"

    @staticmethod
    def view_image(v_img):
        cv2.imwait('image.jpg', v_img)
        cv2.waitKey(0)

    def get_sequence_name(self) -> str:
        return self.sequence_name

    def get_options(self, key=None):
        if not key:
            return self.options
        else:
            return self.options.get(key)

    def get_classes(self):
        return self.classes

    def populate_class_labels(self):
        g: GlobalConfig = self.globs
        if self.options.get('object_labels'):
            labels_file = Path(self.options.get('object_labels'))
            if labels_file.is_file():
                with labels_file.open('r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                g.logger.error(
                    f"{lp} the specified labels file '{labels_file.name}' does not exist/is not a file! "
                    f"Can not populate labels"
                )
                raise YoloException(f"The specified labels file does not exist or is not a file! "
                                    f"({self.options.get('object_labels')})")

    def load_model(self):
        g: GlobalConfig = self.globs
        self.sequence_name = self.options.get('name')
        g.logger.debug(f"{lp} loading model data from sequence '{self.sequence_name}'")
        t = Timer()
        _weights = self.options.get('object_weights')
        _config = self.options.get('object_config')
        if not Path(_weights).is_file() or not Path(_config).is_file():
            raise YoloException(f"The weights or config file does not exist or is not a file!")
        self.net = cv2.dnn.readNet(_weights, _config)
        # self.net = cv2.dnn.readNetFromDarknet(config_file_abs_path, weights_file_abs_path)
        g.logger.debug(
            f"perf:{lp} '{self.sequence_name}' initialization -> loading "
            f"'{Path(_weights).name}' took: {t.stop_and_get_ms()}"
        )
        if self.processor == 'gpu':
            (maj, minor, patch) = cv2.__version__.split('.')
            min_ver = int(maj + minor)
            patch = patch if patch.isnumeric() else 0
            patch_ver = int(maj + minor + patch)
            # 4.5.4 and above (@pliablepixels tracked down the exact version change
            # see https://github.com/ZoneMinder/mlapi/issues/44)
            # @baudneo linked issue -> https://github.com/baudneo/pyzm/issues/3
            if patch_ver >= 454:
                g.logger.info(f"{lp} OpenCV (4.5.4+) fix for getUnconnectedOutLayers() API (Non nested structure)")
                self.new_cv_scalar_fix = True

            if min_ver < 42:
                g.logger.error(
                    f'You are using OpenCV version {cv2.__version__} which does not support CUDA for DNNs. A minimum'
                    f' of 4.2 is required. See https://medium.com/@baudneo/install-zoneminder-1-36-x-6dfab7d7afe7'
                    f' on how to compile and install openCV 4.5.4 with CUDA')
                self.processor = 'cpu'
            else:  # Passed opencv version check, using GPU
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                if str2bool(self.options.get('fp16_target')):
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                    g.logger.debug(
                        f"{lp} half precision floating point (FP16) cuDNN target enabled (turn this off if it"
                        f" makes yolo slower, you will notice if it does!)")
                else:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        g.logger.debug(
            f"{lp} using {self.processor.upper()} for detection"
            f"{', set CUDA/cuDNN backend and target' if self.processor.lower() == 'gpu' else ''}"
        )
        self.populate_class_labels()

    def detect(
            self,
            input_image: Optional[np.ndarray] = None,
            retry: bool = False
    ):
        g: GlobalConfig = self.globs
        if input_image is None:
            g.logger.error(f"{lp} no image passed!?!")
            raise YoloException("NO_IMAGE")
        if not retry:
            self.original_image = deepcopy(input_image)
        indices = None
        ln = None
        blob = None
        outs = None
        downscaled = False
        bbox, label, conf = [], [], []
        upsize_x_factor = None
        upsize_y_factor = None

        h, w = input_image.shape[:2]
        max_size = self.options.get('max_size')
        max_size = w if not max_size else max_size
        if not isinstance(max_size, str) or (isinstance(max_size, str) and max_size.isnumeric()):
            try:
                max_size = int(max_size)
            except TypeError:
                g.logger.error(f"{lp} 'max_size' can only be an integer -> 123 not 123.xx")
                max_size = w
            except Exception as all_ex:
                g.logger.error(
                    f"{lp} ALL EXCEPTION!!! 'max_size' can only be an integer -> 123 not "
                    f"123.xx -> {all_ex}"
                )
                max_size = w
            else:
                if w > max_size:
                    downscaled = True
                    from pyzm.helpers.pyzm_utils import resize_image
                    g.logger.debug(2, f"{lp} scaling image down to max (width) size: {max_size}")
                    input_image = resize_image(input_image, max_size)
                    new_height, new_width = input_image.shape[:2]
                    upsize_x_factor = w / new_width
                    upsize_y_factor = h / new_height

        h, w = input_image.shape[:2]
        try:
            if self.options.get('auto_lock', True):
                self.acquire_lock()
            if not self.net or (self.net and retry):
                # model has not been loaded or this is a retry detection so we want to rebuild
                # the model with changed options.
                self.load_model()
            g.logger.debug(
                f"{lp} '{self.sequence_name}' ({self.processor.upper()}) - input image {w}*{h} - resized by "
                f" model_width/height to: {self.model_width}*{self.model_height}")
            scale = 0.00392  # 1/255, really. Normalize inputs.
            t = Timer()
            ln = self.net.getLayerNames()
            if not self.new_cv_scalar_fix:
                try:
                    ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                except Exception as all_ex:
                    g.logger.error(
                        f"{lp} OpenCV 4.5.4+ 'getUnconnectedOutLayers()' API fix did not work! Please do some testing "
                        f"to see if you have installed the opencv-python pip/pip3 packages by mistake if you compiled "
                        f"OpenCV for GPU/CUDA! see  https://forums.zoneminder.com/viewtopic.php?f=33&t=31293 \n{all_ex}"
                    )
                    raise all_ex
            else:
                ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(
                input_image,
                scale, (self.model_width, self.model_height), (0, 0, 0),
                True,
                crop=False
            )

            self.net.setInput(blob)
            outs = self.net.forward(ln)
        except Exception as all_ex:
            err_msg = repr(all_ex)
            if (
                    err_msg.find('-217:Gpu') > 0
                    and err_msg.find("'make_policy'") > 0
                    and self.options['object_processor'] == 'gpu'
            ):
                g.logger.error(f"{lp} (-217:Gpu # API call) invalid device function in function 'make_policy' - "
                               f"This happens when OpenCV is compiled with the incorrect Compute Capability "
                               f"(CUDA_ARCH_BIN). There is a high probability that you need to recompile OpenCV with "
                               f"the correct CUDA_ARCH_BIN before GPU detections will work properly!")
                # set arch to cpu and retry?
                self.options['object_processor'] = 'cpu'
                g.logger.info(f"{lp} GPU detection failed due to probable incorrect CUDA_ARCH_BIN (Compute Capability)."
                              f" Switching to CPU and retrying detection!")
                self.detect(self.original_image, retry=True)
            g.logger.error(f"{lp} exception during blobFromImage -> {all_ex}")

            # cv2.error: OpenCV(4.2.0) /home/<Someone>/opencv/modules/dnn/src/cuda/execution.hpp:52: error: (-217:Gpu
            # API call) invalid device function in function 'make_policy'
            raise YoloException(f"blobFromImage")

        finally:
            if self.options.get('auto_lock', True):
                self.release_lock()

        class_ids, confidences, boxes = [], [], []
        nms_threshold, conf_threshold = 0.4, 0.2
        # Non Max Suppressive
        if float(self.options.get('object_min_confidence')) < conf_threshold:
            conf_threshold = float(self.options.get('object_min_confidence'))
        try:
            """
            FP16 testing for dnn CUDA_TARGET_FP16
            len(outs) = 3]
            len(out) = 8112]
            len(out) = 2028]
            len(out) = 507]
            print(f"{len(outs)=}")
            """
            for out in outs:
                # print(f"{len(out)=}")
                for detection in out:
                    # FP32 and FP16 have different matrices
                    # g.logger.debug(f"dbg:{lp} {detection[:5]=}")
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    wid = int(detection[2] * w)
                    hei = int(detection[3] * h)
                    x = center_x - wid / 2
                    y = center_y - hei / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, wid, hei])
        except OverflowError as ex:
            if self.processor == 'gpu':
                g.logger.debug(
                    f"GPU needs to be reset? try re downloading YOLO models first, if that doesn't work "
                    f"try -> sudo nvidia-smi -r <- to reset the gpu and restart. "
                    f"Google 'how to kill xorg server' if Xorg is holding you up ("
                    f"hint: ctrl+alt+F1 and once done ctrl+alt+F7)"
                )
                g.logger.error(f"{lp} OverflowError: {ex}")
                raise YoloException(f"GPU RESET")
            raise ex
        except YoloException as v_exc:
            if repr(v_exc) == "cannot convert float NaN to integer":
                if str2bool(self.options.get('fp16_target')):
                    g.logger.error(
                        f"{lp} '{repr(v_exc)}' FP_16 CUDA TARGET is configured. Setting CUDA TARGET to FP_32 and "
                        f"re running detection!"
                    )
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.detect(self.original_image, retried=True)
                else:
                    g.logger.error(f"{lp} 'NaNM' (Not a Number) error but the CUDA TARGET is FP32?!?! FATAL!")
                    raise v_exc
        except Exception as e:
            g.logger.error(f"{lp} EXCEPTION while parsing layer output -->\n{e}")
            raise e

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        diff_time = t.stop_and_get_ms()
        g.logger.debug(2, f"perf:{lp}{self.processor.upper()}: '{self.sequence_name}' detection took: {diff_time}")

        label, bbox, conf, box = self.indice_process(boxes, indices, confidences, self.classes, class_ids,
                                                     self.new_cv_scalar_fix)

        if downscaled:
            g.logger.debug(2, f"{lp} scaling bounding boxes back up using  x={upsize_x_factor} y={upsize_y_factor}")
            bbox = self.downscale(bbox, upsize_x_factor, upsize_y_factor)
        if label:
            g.logger.debug(f"{lp} {label} -- {bbox} -- {conf}")
        else:
            g.logger.debug(f"{lp} no detections to return!")
        self.original_image = None
        return (
            bbox,
            label,
            conf,
            [f"yolo[{self.processor.upper()}]"] * len(label)
        )
