import numpy as np

import pyzm.ml.face_train as train
import face_recognition
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

class Face(Base):
    def __init__(self, logger=None, options={},upsample_times=1, num_jitters=0, model='hog'):
        super().__init__(logger)

        self.logger.Debug(
            1,'Initializing face recognition with model:{} upsample:{}, jitters:{}'
            .format(model, upsample_times, num_jitters))

        self.upsample_times = upsample_times
        self.num_jitters = num_jitters
        self.model = model
        self.knn = None
        self.options = options

        if self.options.get('face_detection_framework') != 'dlib' and self.options.get('face_recognition_framework') != 'dlib':
            raise ValueError ('Error: As of now,only dlib is supported for face detection and recognition. Unkown {}/{}'.format(self.options.get('face_detection_framework'),self.options.get('face_recognition_framework')))

        encoding_file_name = self.options.get('known_images_path') + '/faces.dat'
        try:
            if (os.path.isfile(self.options.get('known_images_path') +
                               '/faces.pickle')):
                # old version, we no longer want it. begone
                self.logger.Debug(
                    1,'removing old faces.pickle, we have moved to clustering')
                os.remove(self.options.get('known_images_path') + '/faces.pickle')
        except Exception as e:
            self.logger.Error('Error deleting old pickle file: {}'.format(e))

        # to increase performance, read encodings from  file
        if (os.path.isfile(encoding_file_name)):
            self.logger.Debug(
                1,'pre-trained faces found, using that. If you want to add new images, remove: {}'
                .format(encoding_file_name))

            #self.known_face_encodings = data["encodings"]
            #self.known_face_names = data["names"]
        else:
            # no encodings, we have to read and train
            self.logger.Debug(
                1,'trained file not found, reading from images and doing training...'
            )
            self.logger.Debug(
                1,'If you are using a GPU and run out of memory, do the training using zm_train_faces.py. In this case, other models like yolo may already take up a lot of GPU memory'
            )

            train.FaceTrain(options=self.options).train()
        try:
            with open(encoding_file_name, 'rb') as f:
                self.knn = pickle.load(f)
        except Exception as e:
            self.logger.Error ('Error loading KNN model: {}'.format(e))

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
        self.logger.Debug(
            1,'|---------- Face recognition (input image: {}w*{}h) ----------|'.
            format(Width, Height))
        labels = []
        classes = []
        conf = []

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = image[:, :, ::-1]
        #rgb_image = image

        # Find all the faces and face encodings in the target image

        start = datetime.datetime.now()
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.model,
            number_of_times_to_upsample=self.upsample_times)

        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(1,'Finding faces took {} milliseconds'.format(diff_time))

        start = datetime.datetime.now()
        face_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters)
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'Computing face recognition distances took {} milliseconds'.format(
                diff_time))

        if not len(face_encodings):
            return [], [], []

        # Use the KNN model to find the best matches for the test face
      
        start = datetime.datetime.now()

        if self.knn:
            closest_distances = self.knn.kneighbors(face_encodings, n_neighbors=1)
            are_matches = [
                closest_distances[0][i][0] <= self.options.get('face_recog_dist_threshold')
                for i in range(len(face_locations))
                
            ]
            prediction_labels = self.knn.predict(face_encodings)

        else:
            # There were no faces to compare
            # create a set of non matches for each face found
            are_matches = [False] * len(face_locations)
            prediction_labels = [''] * len(face_locations)
            self.logger.Debug (1,'No faces to match, so creating empty set')

        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'Matching recognized faces to known faces took {} milliseconds'.
            format(diff_time))

        matched_face_names = []
        matched_face_rects = []

        for pred, loc, rec in zip(prediction_labels,
                                  face_locations, are_matches):
            label = pred if rec else self.options.get('unknown_face_name')
            if not rec and self.options.get('save_unknown_faces') == 'yes':
                h, w, c = image.shape
                x1 = max(loc[3] - self.options.get('save_unknown_faces_leeway_pixels'),
                         0)
                y1 = max(loc[0] - self.options.get('save_unknown_faces_leeway_pixels'),
                         0)

                x2 = min(loc[1] + self.options.get('save_unknown_faces_leeway_pixels'),
                         w)
                y2 = min(loc[2] + self.options.get('save_unknown_faces_leeway_pixels'),
                         h)
                #print (image)
                crop_img = image[y1:y2, x1:x2]
                # crop_img = image
                timestr = time.strftime("%b%d-%Hh%Mm%Ss-")
                unf = self.options.get('unknown_images_path') + '/' + timestr + str(
                    uuid.uuid4()) + '.jpg'
                self.logger.Info(
                    'Saving cropped unknown face at [{},{},{},{} - includes leeway of {}px] to {}'
                    .format(x1, y1, x2, y2,
                            self.options.get('save_unknown_faces_leeway_pixels'), unf))
                cv2.imwrite(unf, crop_img)

            matched_face_rects.append((loc[3], loc[0], loc[1], loc[2]))
            matched_face_names.append(label)
            conf.append(1)

        #self.logger.Debug(1,f'FACE:Returning: {matched_face_rects}, {matched_face_names}, {conf}')
        return matched_face_rects, matched_face_names, conf
