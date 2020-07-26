import numpy as np
import sys
import time
import datetime
import re
import cv2
from pyzm.helpers.Base import Base

from edgetpu.detection.engine import DetectionEngine
from PIL import Image


# Class to handle Yolo based detection


class Tpu(Base):

    def __init__(self, options={},logger=None ):
        Base.__init__(self,logger)
        self.classes = {}
        self.options = options
        start = datetime.datetime.now()
        self.logger.Debug (1,'TPU loading {}'.format(self.options.get('object_weights')))
        self.model = DetectionEngine(self.options.get('object_weights'))
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'TPU initialization (loading model from disk) took: {} milliseconds'
            .format(diff_time))
        self.populate_class_labels()

    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        for row in open(class_file_abs_path):
            # unpack the row and update the labels dictionary
            (classID, label) = row.strip().split(" ", maxsplit=1)
            self.classes[int(classID)] = label.strip()

    def get_classes(self):
        return self.classes


    def detect(self, image=None):

        Height, Width = image.shape[:2]

        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        self.logger.Debug(1,
            '|---------- TPU (input image: {}w*{}h) ----------|'
            .format(Width, Height))
   
       
        start = datetime.datetime.now()
       
        outs = self.model.detect_with_image(img, threshold=self.options.get('object_min_confidence'),
        keep_aspect_ratio=True, relative_coord=False)

        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'Coral TPU detection took: {} milliseconds'.format(diff_time))

        bbox = []
        labels = []
        conf = []

        for out in outs:
            box = out.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            bbox.append([
                    int(round(startX)),
                    int(round(startY)),
                    int(round(endX)),
                    int(round(endY))
                ])
            labels.append(self.classes[out.label_id])
            conf.append(float(out.score))

            
        return bbox, labels, conf
