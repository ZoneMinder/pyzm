# Face wrapper class for recognition, detection is completely separate

from portalocker import AlreadyLocked, BoundedSemaphore

from pyzm.helpers.pyzm_utils import str2bool


class Face:
    def __init__(self, options=None, *args, **kwargs):

        if options is None:
            options = {}
        self.globs = kwargs.get('globs')
        self.model = None
        self.options = options
        name = self.options.get('name') or 'Face wrapper'
        self.lock = None
        self.sequence_name: str = name

        if self.options.get('face_detection_framework') == 'dlib':
            import pyzm.ml.face_dlib as face_dlib
            self.model = face_dlib.FaceDlib(self.options, **kwargs)
        elif self.options.get('face_detection_framework') == 'tpu':
            import pyzm.ml.face_tpu as face_tpu
            self.model = face_tpu.FaceTpu(self.options, **kwargs)
        else:
            raise ValueError(f"{self.options.get('face_detection_framework')} face detection framework is unknown")

    def detect(self, input_image):
        return self.model.detect(input_image)

    def get_options(self):
        return self.model.get_options()

    def acquire_lock(self):
        return self.model.acquire_lock()

    def release_lock(self):
        return self.model.release_lock()

    def load_model(self):
        return

    def get_model_name(self):
        return