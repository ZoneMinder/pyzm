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

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
#from edgetpu.detection.engine import DetectionEngine

# Class to handle Yolo based detection


class Tpu(Base):

    def __init__(self, options={},logger=None ):
        Base.__init__(self,logger)
        self.classes = {}
        self.options = options
       #self.logger.Debug (1, 'UID:{} EUID:{}'.format( os.getuid(), os.geteuid()))

        self.logger.Debug (4, 'TPU init params: {}'.format(options))
        
        self.processor='tpu'
        self.lock_maximum=int(options.get(self.processor+'_max_processes') or 1)
        self.lock_name='pyzm_uid{}_{}_lock'.format(os.getuid(),self.processor)
        self.lock_timeout = int(options.get(self.processor+'_max_lock_wait') or 120)

        self.logger.Debug (2,f'portalock: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}')
        self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
        self.is_locked = False
        self.model = None
        self.populate_class_labels()

    def acquire_lock(self):
        if self.is_locked:
            self.logger.Debug (2, '{} portalock already acquired'.format(self.lock_name))
            return
        try:
            self.logger.Debug (2,'Waiting for {} portalock...'.format(self.lock_name))
            self.lock.acquire()
            self.logger.Debug (2,'Got {} portalock'.format(self.lock_name))
            self.is_locked = True

        except portalocker.AlreadyLocked:
            self.logger.Error ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))

    def release_lock(self):
        if not self.is_locked:
            self.logger.Debug (2, '{} portalock already released'.format(self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        self.logger.Debug (2,'Released portalock {}'.format(self.lock_name))


    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')        
        for row in open(class_file_abs_path):
            # unpack the row and update the labels dictionary
            (classID, label) = row.strip().split(" ", maxsplit=1)
            self.classes[int(classID)] = label.strip()

    def get_classes(self):
        return self.classes

    def load_model(self):
        self.logger.Debug (1, '|--------- Loading TPU model from disk -------------|')
        start = datetime.datetime.now()

        #self.model = DetectionEngine(self.options.get('object_weights'))
        # Initialize the TF interpreter
        self.model = make_interpreter(self.options.get('object_weights'))

        self.model.allocate_tensors()

        diff_time = (datetime.datetime.now() - start)
        self.logger.Debug(
            1,'TPU initialization (loading model from disk) took: {}'
            .format(diff_time))
        
    def detect(self, image=None):
        Height, Width = image.shape[:2]
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.options.get('auto_lock',True):
            self.acquire_lock()

        try:
            if not self.model:
                self.load_model()

            self.logger.Debug(1,
                '|---------- TPU (input image: {}w*{}h) ----------|'
                .format(Width, Height))
            start = datetime.datetime.now()
            
            _, scale = common.set_resized_input(
                self.model, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
            self.model.invoke()
            objs = detect.get_objects(self.model, float(self.options.get('object_min_confidence')), scale)

        
            #outs = self.model.detect_with_image(img, threshold=int(self.options.get('object_min_confidence')),
            #        keep_aspect_ratio=True, relative_coord=False)
            diff_time = (datetime.datetime.now() - start)

            if self.options.get('auto_lock',True):
                self.release_lock()
        except:
            if self.options.get('auto_lock',True):
                self.release_lock()
            raise

        self.logger.Debug(
            1,'Coral TPU detection took: {}'.format(diff_time))
      
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
            labels.append(self.classes[obj.id])
            conf.append(float(obj.score))

        self.logger.Debug(3,'Coral returning: {},{},{}'.format(bbox,labels,conf))
        return bbox, labels, conf
