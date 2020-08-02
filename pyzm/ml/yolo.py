import numpy as np

import sys
import cv2
import time
import datetime
import re
from pyzm.helpers.Base import Base
import portalocker
# Class to handle Yolo based detection


class Yolo(Base):

    # The actual CNN object detection code
    # opencv DNN code credit: https://github.com/arunponnusamy/cvlib

    def __init__(self, options={}, logger=None):
        super().__init__(logger)
        self.net = None
        self.classes = None
        self.options = options

        config_file_abs_path = self.options.get('object_config')
        weights_file_abs_path = self.options.get('object_weights')
        
        self.processor=self.options.get('object_processor') or 'cpu'
        if self.processor == 'gpu':
            (maj, minor, patch) = cv2.__version__.split('.')
            min_ver = int(maj + minor)
            if min_ver < 42:
                self.logger.Error('Not setting CUDA backend for OpenCV DNN')
                self.logger.Error(
                    'You are using OpenCV version {} which does not support CUDA for DNNs. A minimum of 4.2 is required. See https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/ on how to compile and install openCV 4.2'
                    .format(cv2.__version__))
                self.processor = 'cpu'
        else:
            self.logger.Debug (1, 'Using CPU for detection')

        self.logger.Debug(1,'Initializing Yolo')
        self.logger.Debug(2,'config:{}, weights:{}'.format(
            config_file_abs_path, weights_file_abs_path))
        self.populate_class_labels()

       
        self.lock_maximum=options.get(self.processor+'_max_processes') or 1
        self.lock_timeout = options.get(self.processor+'_max_lock_wait') or 120
        
        self.lock_name='pyzm_'+self.processor+'_lock'
        self.logger.Debug (2,f'Semaphore: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}')
        self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
        
        try:
            self.logger.Debug (1,f'Waiting for {self.processor} lock...')
            self.lock.acquire()
            self.logger.Debug (1,f'Got {self.processor} lock for initialization...')
            start = datetime.datetime.now()
            self.net = cv2.dnn.readNet(weights_file_abs_path,
                                    config_file_abs_path)
            #self.net = cv2.dnn.readNetFromDarknet(config_file_abs_path, weights_file_abs_path)
            self.lock.release()
            self.logger.Debug(1,'init lock released')
            diff_time = (datetime.datetime.now() - start).microseconds / 1000
            
            if self.processor == 'gpu':
                self.logger.Debug(
                    1,'Setting CUDA backend for OpenCV. If you did not set your CUDA_ARCH_BIN correctly during OpenCV compilation, you will get errors during detection related to invalid device/make_policy')
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.logger.Debug(
                1,'YOLO initialization (loading model from disk) took: {} milliseconds'
                .format(diff_time))
                
        except portalocker.AlreadyLocked:
            self.logger.Error ('Timeout waiting for {} lock for {} seconds'.format(self.processor, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} lock for {} seconds'.format(self.processor,self.lock_timeout))
    
    
        
    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        f = open(class_file_abs_path, 'r')
        self.classes = [line.strip() for line in f.readlines()]

    def get_classes(self):
        return self.classes

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [
            layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        return output_layers

    def detect(self, image=None):

        Height, Width = image.shape[:2]
        modelW = 416
        modelH = 416

        self.logger.Debug(
            1,'|---------- YOLO (input image: {}w*{}h, resized to: {}w*{}h) ----------|'
            .format(Width, Height, modelW, modelH))
        scale = 0.00392  # 1/255, really. Normalize inputs.

        try:
            self.logger.Debug (1,f'Waiting for {self.processor} detection lock...')
            self.lock.acquire()
            self.logger.Debug (1,f'Got {self.processor} lock for detection')
            start = datetime.datetime.now()
            ln = self.net.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(image,
                                        scale, (modelW, modelH), (0, 0, 0),
                                        True,
                                        crop=False)
            
            self.net.setInput(blob)
            outs = self.net.forward(ln)
            self.lock.release()
            self.logger.Debug(1,'detect lock released')
            diff_time = (datetime.datetime.now() - start).microseconds / 1000
            self.logger.Debug(
                1,'YOLO detection took: {} milliseconds'.format(diff_time))

        except portalocker.AlreadyLocked:
            self.logger.Error ('Timeout waiting for {} lock for {} seconds'.format(self.processor, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} lock for {} seconds'.format(self.processor,self.lock_timeout))
      
        class_ids = []
        confidences = []
        boxes = []

        nms_threshold = 0.4
        conf_threshold = 0.2

        # first nms filter out with a yolo confidence of 0.2 (or less)
        if self.options.get('object_min_confidence') < conf_threshold:
            conf_threshold = self.options.get('object_min_confidence')

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

        start = datetime.datetime.now()
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'YOLO NMS filtering took: {} milliseconds'.format(diff_time))

        bbox = []
        label = []
        conf = []

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
            label.append(str(self.classes[class_ids[i]]))
            conf.append(confidences[i])
           
        
        return bbox, label, conf
