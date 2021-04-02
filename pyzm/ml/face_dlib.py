import numpy as np

import pyzm.ml.face_train_dlib as train

import dlib

import sys
import os
import cv2
import pickle
from sklearn import neighbors
import imutils
import math
import uuid
import time
import datetime
from pyzm.helpers.Base import Base
# Class to handle face recognition
import portalocker
import re
from pyzm.helpers.Media import MediaStream
from pyzm.helpers.utils import Timer
import pyzm.helpers.globals as g

from pyzm.ml.face import Face


g_start =Timer()
import face_recognition
g_diff_time = g_start.stop_and_get_ms()

class FaceDlib(Face):
    def __init__(self, options={}):
        self.options = options
        global g_diff_time
        #g.logger.Debug (4, 'Face init params: {}'.format(options))

        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >=1 :
            self.processor = 'gpu'
        else:
            self.processor = 'cpu'
     
        g.logger.Debug(
            1,'perf: processor:{} Face Recognition library load time took: {} '.format(
                self.processor, g_diff_time))

        upsample_times = self.options.get('upsample_times',1)
        num_jitters= self.options.get('num_jitters',0)
        model=self.options.get('model','hog')

        g.logger.Debug(
            1,'Initializing face recognition with model:{} upsample:{}, jitters:{}'
            .format(model, upsample_times, num_jitters))

        self.disable_locks = options.get('disable_locks', 'no')

        self.upsample_times = upsample_times
        self.num_jitters = num_jitters
        if options.get('face_model'):
            self.face_model = options.get('face_model')
        else:
            self.face_model = model
       
        self.knn = None
        self.options = options
        self.is_locked = False

        self.lock_maximum=int(options.get(self.processor+'_max_processes') or 1)
        self.lock_timeout = int(options.get(self.processor+'_max_lock_wait') or 120)
        
        #self.lock_name='pyzm_'+self.processor+'_lock'
        self.lock_name='pyzm_uid{}_{}_lock'.format(os.getuid(),self.processor)
        if self.disable_locks == 'no':
            g.logger.Debug (2,f'portalock: max:{self.lock_maximum}, name:{self.lock_name}, timeout:{self.lock_timeout}')
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
            
        encoding_file_name = self.options.get('known_images_path') + '/faces.dat'
        try:
            if (os.path.isfile(self.options.get('known_images_path') +
                               '/faces.pickle')):
                # old version, we no longer want it. begone
                g.logger.Debug(
                    1,'removing old faces.pickle, we have moved to clustering')
                os.remove(self.options.get('known_images_path') + '/faces.pickle')
        except Exception as e:
            g.logger.Error('Error deleting old pickle file: {}'.format(e))

        # to increase performance, read encodings from  file
        if (os.path.isfile(encoding_file_name)):
            g.logger.Debug(
                1,'pre-trained faces found, using that. If you want to add new images, remove: {}'
                .format(encoding_file_name))

            #self.known_face_encodings = data["encodings"]
            #self.known_face_names = data["names"]
        else:
            # no encodings, we have to read and train
            g.logger.Debug(
                1,'trained file not found, reading from images and doing training...'
            )
            g.logger.Debug(
                1,'If you are using a GPU and run out of memory, do the training using zm_train_faces.py. In this case, other models like yolo may already take up a lot of GPU memory'
            )

            train.FaceTrain(options=self.options).train()
        try:
            with open(encoding_file_name, 'rb') as f:
                self.knn = pickle.load(f)
                f.close()
        except Exception as e:
            g.logger.Error ('Error loading KNN model: {}'.format(e))


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

    

    def detect(self, image):
      
        Height, Width = image.shape[:2]
        g.logger.Debug(
            1,'|---------- Dlib Face recognition (input image: {}w*{}h) ----------|'.
            format(Width, Height))

        downscaled =  False
        upsize_xfactor = None
        upsize_yfactor = None
        max_size = self.options.get('max_size', Width)
        old_image = None

        g.logger.Debug(5, 'Face options={}'.format(self.options))
        
        if Width > max_size:
            downscaled = True
            g.logger.Debug (2, 'Scaling image down to max size: {}'.format(max_size))
            old_image = image.copy()
            image = imutils.resize(image,width=max_size)
            newHeight, newWidth = image.shape[:2]
            upsize_xfactor = Width/newWidth
            upsize_yfactor = Height/newHeight
        

        labels = []
        classes = []
        conf = []

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = image[:, :, ::-1]
        #rgb_image = image

        # Find all the faces and face encodings in the target image
        #prin (self.options)
        if self.options.get('auto_lock',True):
            self.acquire_lock()

        t = Timer()
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.face_model,
            number_of_times_to_upsample=self.upsample_times)

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(1,'perf: processor:{} Finding faces took {}'.format(self.processor, diff_time))

        t = Timer()
        face_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters)
        
        if self.options.get('auto_lock',True):
            self.release_lock()

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1,'perf: processor:{} Computing face recognition distances took {}'.format(
                self.processor, diff_time))

        if not len(face_encodings):
            return [], [], []

        # Use the KNN model to find the best matches for the test face
      
        g.logger.Debug(3,'Comparing to known faces...')

        t = Timer()
        if self.knn:
            #g.logger.Debug(5, 'FACE ENCODINGS={}'.format(face_encodings))
            closest_distances = self.knn.kneighbors(face_encodings, n_neighbors=1)
            g.logger.Debug(5, 'Closest knn match indexes (lesser is better): {}'.format(closest_distances))
            are_matches = [
                closest_distances[0][i][0] <= float(self.options.get('face_recog_dist_threshold',0.6))
                for i in range(len(face_locations))
                
            ]
            prediction_labels = self.knn.predict(face_encodings)
            #g.logger.Debug(5, 'KNN predictions: {} are_matches: {}'.format(prediction_labels, are_matches))

        else:
            # There were no faces to compare
            # create a set of non matches for each face found
            are_matches = [False] * len(face_locations)
            prediction_labels = [''] * len(face_locations)
            g.logger.Debug (1,'No faces to match, so creating empty set')

        diff_time = t.stop_and_get_ms()
        g.logger.Debug(
            1,'perf: processor:{} Matching recognized faces to known faces took {}'.
            format(self.processor, diff_time))

        matched_face_names = []
        matched_face_rects = []


        if downscaled:
            g.logger.Debug (2,'Scaling image back up to {}'.format(Width))
            image = old_image
            new_face_locations = []
            for loc in face_locations:
                a,b,c,d=loc
                a = round(a * upsize_yfactor)
                b = round(b * upsize_xfactor)
                c = round(c * upsize_yfactor)
                d = round(d * upsize_xfactor)
                new_face_locations.append((a,b,c,d))
            face_locations = new_face_locations


        for pred, loc, rec in zip(prediction_labels,
                                  face_locations, are_matches):
            label = pred if rec else self.options.get('unknown_face_name', 'unknown')
            if not rec and self.options.get('save_unknown_faces') == 'yes':
                h, w, c = image.shape
                x1 = max(loc[3] - int(self.options.get('save_unknown_faces_leeway_pixels',0)),0)
                y1 = max(loc[0] - int(self.options.get('save_unknown_faces_leeway_pixels',0)),0)
                x2 = min(loc[1] + int(self.options.get('save_unknown_faces_leeway_pixels',0)), w)
                y2 = min(loc[2] + int(self.options.get('save_unknown_faces_leeway_pixels',0)),h)
                #print (image)
                crop_img = image[y1:y2, x1:x2]
                # crop_img = image
                timestr = time.strftime("%b%d-%Hh%Mm%Ss-")
                unf = self.options.get('unknown_images_path') + '/' + timestr + str(
                    uuid.uuid4()) + '.jpg'
                g.logger.Info(
                    'Saving cropped unknown face at [{},{},{},{} - includes leeway of {}px] to {}'
                    .format(x1, y1, x2, y2,
                            self.options.get('save_unknown_faces_leeway_pixels'), unf))
                cv2.imwrite(unf, crop_img)

            
            matched_face_rects.append([loc[3], loc[0], loc[1], loc[2]])
            matched_face_names.append(label)
            #matched_face_names.append('face:{}'.format(label))
            conf.append(1)

        g.logger.Debug(3,f'Face Dlib:Returning: {matched_face_rects}, {matched_face_names}, {conf}')
        return matched_face_rects, matched_face_names, conf
