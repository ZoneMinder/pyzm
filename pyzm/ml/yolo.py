import numpy as np

import sys
import cv2
import time
import datetime
import re
from pyzm.helpers.Base import Base
import portalocker
import os
from pyzm.helpers.utils import Timer
import pyzm.helpers.globals as g

# Class to handle Yolo based detection


class Yolo(Base):

    # The actual CNN object detection code
    # opencv DNN code credit: https://github.com/arunponnusamy/cvlib

    def __init__(self, options={}):
        self.net = None
        self.classes = None
        self.options = options
        self.is_locked = False

        #g.logger.Debug (4, 'Yolo init params: {}'.format(options))

        self.processor=self.options.get('object_processor') or 'cpu'
        self.lock_maximum=int(options.get(self.processor+'_max_processes') or 1)
        self.lock_timeout = int(options.get(self.processor+'_max_lock_wait') or 120)
        
        #self.lock_name='pyzm_'+self.processor+'_lock'
        self.lock_name='pyzm_uid{}_{}_lock'.format(os.getuid(),self.processor)

        self.disable_locks = options.get('disable_locks', 'no')
        if self.disable_locks == 'no':
            g.logger.Debug (2,f'portalock: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}')
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
        self.model_height = self.options.get('model_height', 416)
        self.model_width = self.options.get('model_width', 416)

    def acquire_lock(self):
        if self.disable_locks=='yes':
            return
        if self.is_locked:
            g.logger.Debug(2, '{} portalock already acquired'.format(self.lock_name))
            return
        try:
            g.logger.Debug (2,f'Waiting for {self.lock_name} portalock...')
            self.lock.acquire()
            g.logger.Debug (2,f'Got {self.lock_name} portalock')
            self.is_locked = True
           
        except portalocker.AlreadyLocked:
            g.logger.Error ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))


    def release_lock(self):
        if self.disable_locks=='yes':
            return
        if not self.is_locked:
            g.logger.Debug (2, '{} portalock already released'.format(self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        g.logger.Debug (2,'Released {} portalock'.format(self.lock_name))


    def get_options(self):
        return self.options
        
    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        f = open(class_file_abs_path, 'r')
        self.classes = [line.strip() for line in f.readlines()]
        f.close()

    def get_classes(self):
        return self.classes

    def load_model(self):
        name = self.options.get('name') or 'Yolo'
        g.logger.Debug (1, '|--------- Loading "{}" model from disk -------------|'.format(name))
        t = Timer()
        self.net = cv2.dnn.readNet(self.options.get('object_weights'),
                                self.options.get('object_config'))
        #self.net = cv2.dnn.readNetFromDarknet(config_file_abs_path, weights_file_abs_path)
        diff_time = t.stop_and_get_ms()

        g.logger.Debug(
            1,'perf: processor:{} Yolo initialization (loading {} model from disk) took: {}'
            .format(self.processor, self.options.get('object_weights'), diff_time))
        if self.processor == 'gpu':
            (maj, minor, patch) = cv2.__version__.split('.')
            min_ver = int(maj + minor)
            if min_ver < 42:
                g.logger.Error('Not setting CUDA backend for OpenCV DNN')
                g.logger.Error(
                    'You are using OpenCV version {} which does not support CUDA for DNNs. A minimum of 4.2 is required. See https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/ on how to compile and install openCV 4.2'
                    .format(cv2.__version__))
                self.processor = 'cpu'
        else:
            g.logger.Debug (1, 'Using CPU for detection')

        if self.processor == 'gpu':
            g.logger.Debug( 2,'Setting CUDA backend for OpenCV')
            g.logger.Debug( 3,'If you did not set your CUDA_ARCH_BIN correctly during OpenCV compilation, you will get errors during detection related to invalid device/make_policy')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.populate_class_labels()

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [
            layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        return output_layers

    def detect(self, image=None):
        Height, Width = image.shape[:2]
        downscaled =  False
        upsize_xfactor = None
        upsize_yfactor = None
        max_size = self.options.get('max_size', Width)
        old_image = None

        if Width > max_size:
            downscaled = True
            g.logger.Debug (2, 'Scaling image down to max size: {}'.format(max_size))
            old_image = image.copy()
            image = imutils.resize(image,width=max_size)
            newHeight, newWidth = image.shape[:2]
            upsize_xfactor = Width/newWidth
            upsize_yfactor = Height/newHeight        


        if self.options.get('auto_lock',True):
            self.acquire_lock()

        try:
            if not self.net:
                self.load_model()

            g.logger.Debug(
                1,'|---------- YOLO (input image: {}w*{}h, model resize dimensions: {}w*{}h) ----------|'
                .format(Width, Height, self.model_width, self.model_height))

            
            scale = 0.00392  # 1/255, really. Normalize inputs.
                
            t = Timer()
            ln = self.net.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(image,
                                        scale, (self.model_width, self.model_height), (0, 0, 0),
                                        True,
                                        crop=False)
            
            self.net.setInput(blob)
            outs = self.net.forward(ln)

            if self.options.get('auto_lock',True):
                self.release_lock()
        except:
            if self.options.get('auto_lock',True):
                self.release_lock()
            raise

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1,'perf: processor:{} Yolo detection took: {}'.format(self.processor, diff_time))

    
        class_ids = []
        confidences = []
        boxes = []

        nms_threshold = 0.4
        conf_threshold = 0.2

        # first nms filter out with a yolo confidence of 0.2 (or less)
        if float(self.options.get('object_min_confidence')) < conf_threshold:
            conf_threshold = float(self.options.get('object_min_confidence'))

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

        t = Timer()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        diff_time = t.stop_and_get_ms()

        g.logger.Debug(
            2,'perf: processor:{} Yolo NMS filtering took: {}'.format(self.processor, diff_time))

        bbox = []
        label = []
        conf = []

        prefix = '(yolo) ' if self.options.get('show_models')=='yes' else ''

        # now filter out with configured yolo confidence, so we can see rejections in log
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            
            bbox.append([
                int(round(x)),
                int(round(y)),
                int(round(x + w)),
                int(round(y + h))
            ])
            label.append(prefix+str(self.classes[class_ids[i]]))
            conf.append(confidences[i])
           
        if downscaled:
            g.logger.Debug (2,'Scaling image back up to {}'.format(Width))
            image = old_image
            for box in bbox:
                box[0] = round (box[0] * upsize_xfactor)
                box[1] = round (box[1] * upsize_yfactor)
                box[2] = round (box[2] * upsize_xfactor)
                box[3] = round (box[3] * upsize_yfactor)
        
        return bbox, label, conf
