
from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g




class Face(Base):
    def __init__(self, options={}):

        self.model = None
        self.options = options
        if self.options.get('face_detection_framework') == 'dlib':
            import pyzm.ml.face_dlib as face_dlib
            self.model = face_dlib.FaceDlib(self.options)
        elif self.options.get('face_detection_framework') == 'tpu':
            import pyzm.ml.face_tpu as face_tpu
            self.model = face_tpu.FaceTpu(self.options)
        else:
            raise ValueError ('{} face detection framework is unknown'.format(self.options.get('face_detection_framework')))
   
    def detect(self, image):
        return self.model.detect(image)
    
    def get_options(self):
        return self.model.get_options()
        