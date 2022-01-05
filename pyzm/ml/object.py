from typing import Optional

from portalocker import AlreadyLocked, BoundedSemaphore

from pyzm.helpers.pyzm_utils import str2bool
from pyzm.interface import GlobalConfig

g: GlobalConfig


class Object:
    """'Object' is a BASE class to wrap other model Classes for detections using OpenCV 4.2+/CUDA/cuDNN"""
    def __init__(self, *args, **kwargs):
        global g
        g = GlobalConfig()
        self.lock: Optional[BoundedSemaphore] = None

    def create_lock(self):
        if not str2bool(self.disable_locks):
            g.logger.debug(2,
                           f"{self.lp}portalock: [name: {self.lock_name}] [max: {self.lock_maximum}] - "
                           f"[timeout: {self.lock_timeout}]"
                           )
            self.lock = BoundedSemaphore(
                maximum=self.lock_maximum,
                name=self.lock_name,
                timeout=self.lock_timeout
            )
        else:
            self.lock = None

    def acquire_lock(self):
        if str2bool(self.disable_locks):
            return
        if self.is_locked:
            g.logger.debug(2, f"{self.lp}portalock: '{self.lock_name}' already acquired")
            return
        try:
            g.logger.debug(2, f"{self.lp}portalock: Waiting for '{self.lock_name}' portalock...")
            if self.lock:
                self.lock.acquire()
                g.logger.debug(2, f"{self.lp}portalock: got '{self.lock_name}'")
                self.is_locked = True

        except AlreadyLocked:
            g.logger.error(
                f"{self.lp}portalock: timeout waiting for '{self.lock_name}'  for {self.lock_timeout}"
                f" seconds"
            )
            raise ValueError(
                f'Timeout waiting for {self.lock_name} portalock for {self.lock_timeout} seconds')

    def release_lock(self):
        if str2bool(self.disable_locks):
            return
        if not self.is_locked:
            g.logger.debug(2, f"{self.lp}portalock: already released '{self.lock_name}'")
            return
        if self.lock:
            self.lock.release()
            self.is_locked = False
            g.logger.debug(2, f"{self.lp}portalock: released '{self.lock_name}'")

    @staticmethod
    def downscale(bbox: list, upsize_x_factor: float, upsize_y_factor: float) -> list:
        for box in bbox:
            box[0] = round(box[0] * upsize_x_factor)
            box[1] = round(box[1] * upsize_y_factor)
            box[2] = round(box[2] * upsize_x_factor)
            box[3] = round(box[3] * upsize_y_factor)
        return bbox

    @staticmethod
    def indice_process(
            boxes,
            indices,
            confidences,
            classes,
            class_ids,
            scalar_fix: bool = False
    ):
        box = None
        bbox, label, conf = [], [], []
        for i in indices:
            if not scalar_fix:
                # Nested on versions older than 4.5.4 GetUnconnectedOutLayers() API changed
                i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w_ = box[2]
            h_ = box[3]
            bbox.append([
                int(round(x)),
                int(round(y)),
                int(round(x + w_)),
                int(round(y + h_))
            ])
            label.append(str(classes[class_ids[i]]))
            conf.append(confidences[i])
        return label, bbox, conf, box
