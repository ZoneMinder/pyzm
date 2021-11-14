import cv2
import pickle
from sklearn import neighbors
import imutils
import math
import os
from pathlib import Path

from pyzm.helpers.pyzm_utils import Timer
import face_recognition as face_rec_libs

g = None


class FaceTrain:

    def __init__(self, globs=None):
        global g
        if globs:
            g = globs

        self.options = g.config

    def train(self, size=None):
        t = Timer()
        known_images_path = self.options.get('known_images_path')
        train_model = self.options.get('face_train_model')
        knn_algo = self.options.get('face_recog_knn_algo', 'ball_tree')

        upsample_times = int(self.options.get('face_upsample_times', 1))
        num_jitters = int(self.options.get('face_num_jitters', 3))

        encoding_file_name = f'{known_images_path}/faces.dat'
        try:
            if Path(f'{known_images_path}/faces.pickle').is_file():
                # old version, we no longer want it. begone
                g.logger.debug(
                    2, 'mlapi:face-train: removing old faces.pickle, we have moved to clustering')
                os.remove(f'{known_images_path}/faces.pickle')
        except Exception as e:
            g.logger.error(f"mlapi:face-train: Error deleting old pickle file: {e}")

        directory = known_images_path
        ext = ('.jpg', '.jpeg', '.png', '.gif')
        known_face_encodings = []
        known_face_names = []

        try:
            for entry in os.listdir(directory):
                if Path(f'{directory}/{entry}').is_dir():
                    # multiple images for this person,
                    # so we need to iterate that subdir
                    g.logger.debug(
                        f"mlapi:face-train: '{entry}' is a directory. Processing all images inside it")
                    person_dir = os.listdir(directory + '/' + entry)
                    for person in person_dir:
                        if person.endswith(tuple(ext)):
                            g.logger.debug(f'mlapi:face-train: loading face from  {entry}/{person}')

                            # imread seems to do a better job of color space conversion and orientation
                            known_face = cv2.imread(f'{directory}/{entry}/{person}')
                            if known_face is None or known_face.size == 0:
                                g.logger.error('mlapi:face-train: Error reading file, skipping')
                                continue
                            # known_face = face_recognition.load_image_file('{}/{}/{}'.format(directory,entry, person))
                            if not size:
                                if self.options.get('resize') == 'no':
                                    size = 800
                                else:
                                    size = int(self.options.get('resize', 800))
                            g.logger.debug(f'mlapi:face-train: resizing to {size}')
                            known_face = imutils.resize(known_face, width=size)

                            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                            # g.logger.Debug(1,'Converting from BGR to RGB')
                            known_face = known_face[:, :, ::-1]
                            face_locations = face_rec_libs.face_locations(
                                known_face,
                                model=train_model,
                                number_of_times_to_upsample=upsample_times)
                            if len(face_locations) != 1:
                                g.logger.error(
                                    f"mlapi:face-train: File {person} has {len(face_locations)} faces, cannot use for training. We need exactly 1 face. If you think you have only 1 face try using \"cnn\" for training mode. Ignoring...")
                            else:
                                face_encodings = face_rec_libs.face_encodings(
                                    known_face,
                                    known_face_locations=face_locations,
                                    num_jitters=num_jitters)
                                known_face_encodings.append(face_encodings[0])
                                known_face_names.append(entry)
                                # g.logger.Debug ('Adding image:{} as known person: {}'.format(person, person_dir))

                elif entry.endswith(tuple(ext)):
                    # this was old style. Lets still support it. The image is a single file with no directory
                    g.logger.debug(f"mlapi:face-train: loading face from {entry}")
                    # known_face = cv2.imread('{}/{}/{}'.format(directory,entry, person))
                    known_face = cv2.imread(f"{directory}/{entry}")

                    if not size:
                        if g.config.get('resize') == 'no':
                            size = 800
                        else:
                            size = int(self.options.get('resize', 800))
                        g.logger.debug(f"mlapi:face-train: resizing to {size}")
                        known_face = imutils.resize(known_face, width=size)
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    known_face = known_face[:, :, ::-1]
                    face_locations = face_rec_libs.face_locations(
                        known_face,
                        model=train_model,
                        number_of_times_to_upsample=upsample_times)

                    if len(face_locations) != 1:
                        g.logger.error(
                            f"mlapi:face-train: File {entry} has {len(face_locations)} faces, cannot use for training."
                            f" We need exactly 1 face. If you think you have only 1 face try using 'cnn' for "
                            f"training mode. Ignoring...")
                    else:
                        face_encodings = face_rec_libs.face_encodings(
                            known_face,
                            known_face_locations=face_locations,
                            num_jitters=num_jitters)
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(os.path.splitext(entry)[0])

        except Exception as e:
            g.logger.error(f"mlapi:face-train: Error initializing face recognition: {e}")
            raise ValueError('Error opening known faces directory. Is the path correct?')

        # Now we've finished iterating all files/dirs
        # lets create the svm
        if not len(known_face_names):
            g.logger.error(
                'mlapi:face-train: No known faces found to train, skipping saving of face encodings to file...')
        else:
            n_neighbors = int(
                round(
                    math.sqrt(
                        len(known_face_names)
                    )
                )
            )
            g.logger.debug(2, f"mlapi:face-train: using algo:{knn_algo} n_neighbors to be: {n_neighbors}")
            knn = neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors,
                algorithm=knn_algo,
                weights='distance'
            )

            g.logger.debug('mlapi:face-train: training model ...')
            knn.fit(known_face_encodings, known_face_names)

            try:
                with open(encoding_file_name, "wb") as f:
                    pickle.dump(knn, f)
            except Exception as exc:
                g.logger.error(f"mlapi:face-train: error writing face encodings to pickle file!")
            else:
                g.logger.debug(f"mlapi:face-train: wrote encoding file: {encoding_file_name}")
        diff_time = t.stop_and_get_ms()
        g.logger.debug(f"perf: Face Recognition training took: {diff_time}")
