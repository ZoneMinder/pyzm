import numpy as np

import sys
import cv2
from imutils.object_detection import non_max_suppression
from pyzm.helpers.Base import Base

# Class to handle HOG based detection


class Hog(Base):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.winStride = self.options.get('stride')
        self.padding = self.options.get('padding')
        self.scale = float(self.options.get('scale'))
        self.meanShift = True if int(self.options.get('mean_shift')) > 0 else False
        self.logger.Debug(1,'Initializing HOG')

    def get_classes(self):
        return ['person']

    def detect(self, image):
        r, w = self.hog.detectMultiScale(image,
                                         winStride=self.winStride,
                                         padding=self.padding,
                                         scale=self.scale,
                                         useMeanshiftGrouping=self.meanShift)
        labels = []
        classes = []
        conf = []
        rects = []

        for i in r:
            labels.append('person')
            classes.append('person')
            conf.append(1.0)
            i = i.tolist()
            (x1,y1,x2,y2) = (round(i[0]),round(i[1]),round(i[0]+i[2]), round(i[1]+i[3]))
            rects.append((x1,y1,x2,y2))

        #self.logger.Debug(f'HOG:Returning: {rects}, {labels}, {conf}')
        return rects, labels, conf
