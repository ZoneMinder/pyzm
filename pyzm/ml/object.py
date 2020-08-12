import numpy as np

import sys
import cv2
import time
import datetime
import re
from pyzm.helpers.Base import Base

# Class to handle Yolo based detection




class Object(Base):

    
    def __init__(self, options={}, logger=None):

        Base.__init__(self,logger)
        self.model = None
        self.options = options

        if self.options.get('object_framework') == 'opencv':
            import pyzm.ml.yolo as yolo
            self.model =  yolo.Yolo(options=options, logger=logger)
            

        elif self.options.get('object_framework') == 'coral_edgetpu':
            import pyzm.ml.coral_edgetpu as tpu
            self.model = tpu.Tpu(options=options, logger=logger)

        else:
            raise ValueError ('Invalid object_framework:{}'.format(self.options.get('object_framework')))

    def get_model(self):
            return self.model

    def get_classes(self):
            return self.model.get_classes()

    def detect(self,image=None):
        h,w = image.shape[:2]
        b,l,c = self.model.detect(image)
        self.logger.Debug (2,'core model detection over, got {} objects. Now filtering'.format(len(b)))
        # Apply various object filtering rules
        max_object_area = 0
        if self.options.get('max_detection_size'):
                self.logger.Debug(3,'Max object size found to be: {}'.format(self.options.get('max_detection_size')))
                # Let's make sure its the right size
                m = re.match('(\d*\.?\d*)(px|%)?$', self.options.get('max_detection_size'),
                            re.IGNORECASE)
                if m:
                    max_object_area = float(m.group(1))
                    if m.group(2) == '%':
                        max_object_area = float(m.group(1))/100.0*(h * w)
                        self.logger.Debug (2,'Converted {}% to {}'.format(m.group(1), max_object_area))
                else:
                    self.logger.Error('max_object_area misformatted: {} - ignoring'.format(
                        self.options.get('max_object_area')))

        boxes=[]
        labels=[]
        confidences=[]

        for idx,box in enumerate(b):
            (sX,sY,eX,eY) = box
            if max_object_area:
                object_area = abs((eX-sX)*(eY-sY))
                if (object_area > max_object_area):
                    self.logger.Debug (1,'Ignoring object:{}, as it\'s area: {}px exceeds max_object_area of {}px'.format(l[idx], object_area, max_object_area))
                    continue
            if c[idx] >= self.options.get('object_min_confidence'):
                boxes.append([sX,sY,eX,eY])
                labels.append(l[idx])
                confidences.append(c[idx])
            else:
                self.logger.Debug (1,'Ignoring {} {} as conf. level {} is lower than {}'.format(l[idx],box,c[idx],self.options.get('object_min_confidence')))
       
        self.logger.Debug (2,'Returning filtered list of {} objects.'.format(len(boxes)))
        return boxes,labels,confidences
