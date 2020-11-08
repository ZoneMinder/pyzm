import numpy as np
import sys
import time
import datetime
import re
import cv2
from pyzm.helpers.Base import Base


from PIL import Image
import portalocker
import os

from edgetpu.detection.engine import DetectionEngine

# Class to handle Yolo based detection


class Tpu(Base):

    def __init__(self, options={},logger=None ):
        Base.__init__(self,logger)
        self.classes = {}
        self.options = options
       #self.logger.Debug (1, 'UID:{} EUID:{}'.format( os.getuid(), os.geteuid()))
        
        self.processor='tpu'
        self.lock_maximum=options.get(self.processor+'_max_processes') or 1
        self.lock_name='pyzm_'+self.processor+'_lock'
        self.lock_timeout = options.get(self.processor+'_max_lock_wait') or 120

        self.logger.Debug (2,f'Semaphore: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}')
        self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
        self.is_locked = False
        self.model = None
        self.populate_class_labels()

    def acquire_lock(self):
        if self.is_locked:
            self.logger.Debug (1, '{} Lock already acquired'.format(self.lock_name))
            return
        try:
            self.logger.Debug (1,'Waiting for TPU lock...')
            self.lock.acquire()
            self.logger.Debug (1,'Got TPU Lock')
            self.is_locked = True

        except portalocker.AlreadyLocked:
            self.logger.Error ('Timeout waiting for {} lock for {} seconds'.format(self.processor, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} lock for {} seconds'.format(self.processor, self.lock_timeout))

    def release_lock(self):
        if not self.is_locked:
            self.logger.Debug (1, '{} Lock already released'.format(self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        self.logger.Debug (1,'Released TPU lock')


    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        for row in open(class_file_abs_path):
            # unpack the row and update the labels dictionary
            (classID, label) = row.strip().split(" ", maxsplit=1)
            self.classes[int(classID)] = label.strip()

    def get_classes(self):
        return self.classes

    def load_model(self):
        self.logger.Debug (1, 'Loading TPU model from disk')
        start = datetime.datetime.now()
        self.model = DetectionEngine(self.options.get('object_weights'))
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'TPU initialization (loading model from disk) took: {} milliseconds'
            .format(diff_time))
        
    def detect(self, image=None):
        Height, Width = image.shape[:2]
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.options.get('auto_lock',True):
            self.acquire_lock()

        if not self.model:
           self.load_model()

        self.logger.Debug(1,
            '|---------- TPU (input image: {}w*{}h) ----------|'
            .format(Width, Height))
        start = datetime.datetime.now()
        outs = self.model.detect_with_image(img, threshold=int(self.options.get('object_min_confidence')),
                keep_aspect_ratio=True, relative_coord=False)
        diff_time = (datetime.datetime.now() - start).microseconds / 1000

        if self.options.get('auto_lock',True):
            self.release_lock()

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
