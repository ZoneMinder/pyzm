
import cv2
import argparse
import pickle
from sklearn import neighbors
import imutils
import math
import ssl
import os
import datetime

from pyzm.helpers.Base import Base
from pyzm.helpers.utils import Timer

g_start = datetime.datetime.now()
import face_recognition
g_diff_time = (datetime.datetime.now() - g_start)

class FaceTrain (Base):

    def __init__(self,logger=None, options={}):
        super().__init__(logger)
        global g_diff_time

        self.options = options

    def train(self,size=None):
        t = Timer()
        known_images_path = self.options.get('known_images_path')
        train_model = self.options.get('face_train_model')
        knn_algo = self.options.get('face_recog_knn_algo', 'ball_tree') 
    
        upsample_times = int(self.options.get('face_upsample_times',1))
        num_jitters = int(self.options.get('face_num_jitters',0))

        encoding_file_name = known_images_path + '/faces.dat'
        try:
            if (os.path.isfile(known_images_path + '/faces.pickle')):
                # old version, we no longer want it. begone
                self.logger.Debug(
                    2,'removing old faces.pickle, we have moved to clustering')
                os.remove(known_images_path + '/faces.pickle')
        except Exception as e:
            self.logger.Error('Error deleting old pickle file: {}'.format(e))

        directory = known_images_path
        ext = ['.jpg', '.jpeg', '.png', '.gif']
        known_face_encodings = []
        known_face_names = []

        try:
            for entry in os.listdir(directory):
                if os.path.isdir(directory + '/' + entry):
                    # multiple images for this person,
                    # so we need to iterate that subdir
                    self.logger.Debug(
                        1,'{} is a directory. Processing all images inside it'.
                        format(entry))
                    person_dir = os.listdir(directory + '/' + entry)
                    for person in person_dir:
                        if person.endswith(tuple(ext)):
                            self.logger.Debug(1,'loading face from  {}/{}'.format(
                                entry, person))

                            # imread seems to do a better job of color space conversion and orientation
                            known_face = cv2.imread('{}/{}/{}'.format(
                                directory, entry, person))
                            if known_face is None or known_face.size == 0:
                                self.logger.Error('Error reading file, skipping')
                                continue
                            #known_face = face_recognition.load_image_file('{}/{}/{}'.format(directory,entry, person))
                            if not size:
                                size = int(self.options.get('resize',800))
                            self.logger.Debug (1,'resizing to {}'.format(size))
                            known_face = imutils.resize(known_face,width=size)

                            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                            #self.logger.Debug(1,'Converting from BGR to RGB')
                            known_face = known_face[:, :, ::-1]
                            face_locations = face_recognition.face_locations(
                                known_face,
                                model=train_model,
                                number_of_times_to_upsample=upsample_times)
                            if len(face_locations) != 1:
                                self.logger.Error(
                                    'File {} has {} faces, cannot use for training. We need exactly 1 face. If you think you have only 1 face try using "cnn" for training mode. Ignoring...'
                                .format(person, len(face_locations)))
                            else:
                                face_encodings = face_recognition.face_encodings(
                                    known_face,
                                    known_face_locations=face_locations,
                                    num_jitters=num_jitters)
                                known_face_encodings.append(face_encodings[0])
                                known_face_names.append(entry)
                                #self.logger.Debug ('Adding image:{} as known person: {}'.format(person, person_dir))

                elif entry.endswith(tuple(ext)):
                    # this was old style. Lets still support it. The image is a single file with no directory
                    self.logger.Debug(1,'loading face from  {}'.format(entry))
                    #known_face = cv2.imread('{}/{}/{}'.format(directory,entry, person))
                    known_face = cv2.imread('{}/{}'.format(directory, entry))

                    if not size:
                        size = int(self.options.get('resize',800))
                        self.logger.Debug (1,'resizing to {}'.format(size))
                        known_face = imutils.resize(known_face,width=size)
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    known_face = known_face[:, :, ::-1]
                    face_locations = face_recognition.face_locations(
                        known_face,
                        model=train_model,
                        number_of_times_to_upsample=upsample_times)

                    if len(face_locations) != 1:
                        self.logger.Error(
                                    'File {} has {} faces, cannot use for training. We need exactly 1 face. If you think you have only 1 face try using "cnn" for training mode. Ignoring...'
                                    .format(person), len(face_locations))
                    else:
                        face_encodings = face_recognition.face_encodings(
                            known_face,
                            known_face_locations=face_locations,
                            num_jitters=num_jitters)
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(os.path.splitext(entry)[0])

        except Exception as e:
            self.logger.Error('Error initializing face recognition: {}'.format(e))
            raise ValueError(
                'Error opening known faces directory. Is the path correct?')

        # Now we've finished iterating all files/dirs
        # lets create the svm
        if not len(known_face_names):
            self.logger.Error(
                'No known faces found to train, encoding file not created')
        else:
            n_neighbors = int(round(math.sqrt(len(known_face_names))))
            self.logger.Debug(2,'Using algo:{} n_neighbors to be: {}'.format(knn_algo, n_neighbors))
            knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                algorithm=knn_algo,
                                                weights='distance')

            self.logger.Debug(1,'Training model ...')
            knn.fit(known_face_encodings, known_face_names)

            f = open(encoding_file_name, "wb")
            pickle.dump(knn, f)
            f.close()
            self.logger.Debug(1,'wrote encoding file: {}'.format(encoding_file_name))
        diff_time = t.stop_and_get_ms()
        self.logger.Debug(
            1,'perf: Face Recognition training took: {}'.format(diff_time))
