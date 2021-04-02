import numpy as np


import sys
import os
import cv2

import math
import uuid
import time
import datetime
from pyzm.helpers.Base import Base
# Class to handle face recognition
import portalocker
import re
from pyzm.helpers.Media import MediaStream
import imutils
from pyzm.helpers.utils import Timer
import pyzm.helpers.globals as g
from pyzm.ml.face import Face
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

class FaceTpu(Face):
    def __init__(self, options={}):
        global g_diff_time
        #g.logger.Debug (4, 'Face init params: {}'.format(options))
        self.options = options
      
        g.logger.Debug(
            1,'Initializing face detection')

        self.processor='tpu'
        self.lock_maximum=int(options.get(self.processor+'_max_processes') or 1)
        self.lock_name='pyzm_uid{}_{}_lock'.format(os.getuid(),self.processor)
        self.lock_timeout = int(options.get(self.processor+'_max_lock_wait') or 120)
        self.disable_locks = options.get('disable_locks', 'no')
        if self.disable_locks == 'no':
            g.logger.Debug (2,f'portalock: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}')
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
        self.is_locked = False
        self.model = None


    def get_options(self):
        return self.options
        
    def acquire_lock(self):
        if self.disable_locks=='yes':
            return
        if self.is_locked:
            g.logger.Debug (2, '{} portalock already acquired'.format(self.lock_name))
            return
        try:
            g.logger.Debug (2,f'Waiting for {self.lock_name} portalock...')
            self.lock.acquire()
            g.logger.Debug (2,f'Got {self.lock_name} lock...')
            self.is_locked = True

        except portalocker.AlreadyLocked:
            g.logger.Error ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))


    def release_lock(self):
        if self.disable_locks=='yes':
            return
        if not self.is_locked:
            g.logger.Debug (1, '{} portalock already released'.format(self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        g.logger.Debug (1,'Released {} portalock'.format(self.lock_name))

    def get_classes(self):
        if self.knn:
            return self.knn.classes_
        else:
            return []

    def _rescale_rects(self, a):
        rects = []
        for (left, top, right, bottom) in a:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            rects.append([left, top, right, bottom])
        return rects

    def load_model(self):
        name = self.options.get('name') or 'TPU'
        g.logger.Debug (1, '|--------- Loading "{}" model from disk -------------|'.format(name))

        t = Timer()
        self.model = make_interpreter(self.options.get('face_weights'))
        self.model.allocate_tensors()
        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1,'perf: processor:{} TPU initialization (loading {} from disk) took: {}'
            .format(self.processor, self.options.get('face_weights'),diff_time))
        
    
    

    def detect(self, image):
        Height, Width = image.shape[:2]
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.options.get('auto_lock',True):
            self.acquire_lock()

        try:
            if not self.model:
                self.load_model()

            g.logger.Debug(1,
                '|---------- TPU (input image: {}w*{}h) ----------|'
                .format(Width, Height))
            t= Timer()            
            _, scale = common.set_resized_input(
                self.model, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
            self.model.invoke()
            objs = detect.get_objects(self.model, float(self.options.get('face_min_confidence',0.1)), scale)

        
            #outs = self.model.detect_with_image(img, threshold=int(self.options.get('object_min_confidence')),
            #        keep_aspect_ratio=True, relative_coord=False)
            diff_time = t.stop_and_get_ms()

            if self.options.get('auto_lock',True):
                self.release_lock()
        except:
            if self.options.get('auto_lock',True):
                self.release_lock()
            raise

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1,'perf: processor:{} Coral TPU detection took: {}'.format(self.processor, diff_time))
    
        bbox = []
        labels = []
        conf = []

        prefix = '(coral) ' if self.options.get('show_models')=='yes' else ''
        for obj in objs:
        # box = obj.bbox.flatten().astype("int")
            bbox.append([
                    int(round(obj.bbox.xmin)),
                    int(round(obj.bbox.ymin)),
                    int(round(obj.bbox.xmax)),
                    int(round(obj.bbox.ymax))
                ])
        
            labels.append(prefix+self.options.get('unknown_face_name', 'face'))
            conf.append(float(obj.score))
        g.logger.Debug (4, 'Coral face is detection only. Skipping recognition phase')
        g.logger.Debug(3,'Coral face returning: {},{},{}'.format(bbox,labels,conf))
        return bbox, labels, conf

