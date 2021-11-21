import os
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import dlib
import portalocker

from pyzm.helpers.pyzm_utils import Timer, id_generator, str2bool, resize_image
from pyzm.ml.face import Face

# This is for the remove globals branch
# from pyzm.helpers.new_yaml import GlobalConfig
# g: Optional[GlobalConfig] = None

g = None
g_start = Timer()
import face_recognition

g_diff_time = g_start.stop_and_get_ms()
face_rec_libs = face_recognition
lp = 'dlib:face:'


# Class to handle face recognition
class FaceDlib(Face):
    def __init__(self, options=None, globs=None):
        global g
        if globs:
            g = globs
        if options is None:
            options = {}
        self.options = options
        self.sequence_name: str = self.options.get('name')
        g.logger.debug(4, f"{lp} init params: {options}")

        self.processor = options.get('face_dlib_processor', 'gpu')
        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1 and self.processor == 'gpu':
            g.logger.debug(f"{lp} dlib was compiled with CUDA support and there is an available GPU "
                           f"to use for processing! (Total GPUs dlib could use: {dlib.cuda.get_num_devices()})")
            self.processor = 'gpu'
        elif dlib.DLIB_USE_CUDA and not dlib.cuda.get_num_devices() >= 1 and self.processor == 'gpu':
            g.logger.error(f"{lp} It appears dlib was compiled with CUDA support but there is not an available GPU "
                           f"for dlib to use! Using CPU for dlib detections...")
            self.processor = 'cpu'
        elif not dlib.DLIB_USE_CUDA and self.processor == 'gpu':
            g.logger.error(f"{lp} It appears dlib was not compiled with CUDA support! Using CPU for dlib detections...")
            self.processor = 'cpu'
        else:
            self.processor = 'cpu'

        g.logger.debug(
            f"perf:{lp}{self.processor}: importing Face Recognition library took: {g_diff_time} ")

        self.upsample_times = int(self.options.get('face_upsample_times', 1))
        self.num_jitters = int(self.options.get('face_num_jitters', 0))
        model = self.options.get('face_model', 'hog')

        g.logger.debug(
            f"{lp} initializing face_recognition with DNN model: '{model}' upsample_times: {self.upsample_times},"
            f" num_jitters (distort): {self.num_jitters}"
        )

        self.disable_locks = options.get('disable_locks', 'no')
        if options.get('face_model'):
            self.face_model = options.get('face_model')
        else:
            self.face_model = model

        self.knn = None
        self.options = options
        self.is_locked = False

        self.lock_maximum = int(options.get(self.processor + '_max_processes') or 10)
        self.lock_timeout = int(options.get(self.processor + '_max_lock_wait') or 220)

        # self.lock_name='pyzm_'+self.processor+'_lock'
        self.lock_name = f"pyzm_uid{os.getuid()}_{self.processor.upper()}_lock"
        if not str2bool(self.disable_locks):
            g.logger.debug(2,
                           f"{lp}portalock: [max: {self.lock_maximum}] [name: {self.lock_name}] "
                           f"[timeout: {self.lock_timeout}]"
                           )
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,
                                                     timeout=self.lock_timeout)

        encoding_file_name = f"{self.options.get('known_images_path')}/faces.dat"
        # to increase performance, read encodings from file
        if Path(encoding_file_name).is_file():
            g.logger.debug(
                f"{lp} pre-trained (known) faces found. If you want to add new images, "
                f"remove: '{encoding_file_name}'"
            )
        try:
            with Path(encoding_file_name).open('rb') as f:
                self.knn = pickle.load(f)
        except Exception as e:
            g.logger.error(f"{lp} error loading KNN model from faces.dat -> {e}")
            return

    def get_options(self):
        return self.options

    def acquire_lock(self):
        if str2bool(self.disable_locks):
            return
        if self.is_locked:
            g.logger.debug(2, f"{lp}portalock: already acquired -> '{self.lock_name}'")
            return
        try:
            g.logger.debug(2, f"{lp}portalock: Waiting for '{self.lock_name}'")
            self.lock.acquire()
            g.logger.debug(2, f"{lp}portalock: acquired -> '{self.lock_name}'")
            self.is_locked = True

        except portalocker.AlreadyLocked:
            g.logger.error(f"{lp}portalock: Timeout waiting for -> '{self.lock_timeout}' sec: {self.lock_name}")
            raise ValueError(f"portalock: Timeout waiting for {self.lock_timeout} sec: {self.lock_name}")

    def release_lock(self):
        if str2bool(self.disable_locks):
            return
        if not self.is_locked:
            # g.logger.debug(2, f"portalock: already released: {self.lock_name}")
            return
        self.lock.release()
        self.is_locked = False
        g.logger.debug(2, f"{lp}portalock: released -> '{self.lock_name}'")

    def get_classes(self):
        if self.knn:
            return self.knn.classes_
        else:
            return []

    @staticmethod
    def _rescale_rects(a):
        rects = []
        for (left, top, right, bottom) in a:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            rects.append([left, top, right, bottom])
        return rects

    @property
    def get_model_name(self) -> str:
        return 'Face-Dlib'

    @property
    def get_sequence_name(self) -> str:
        return self.sequence_name

    def detect(self, input_image):
        # global face_rec_libs
        detect_start_timer = Timer()
        Height, Width = input_image.shape[:2]

        downscaled = False
        upsize_x_factor = None
        upsize_y_factor = None
        max_size = self.options.get('max_size', Width)
        old_image = None
        newWidth, newHeight = None, None
        # g.logger.debug(5, f"{lp} options={self.options}")

        if Width > int(max_size):
            downscaled = True
            g.logger.debug(2, f"{lp} scaling image down using 'max_size' as width: {max_size}")
            old_image = input_image.copy()
            input_image = resize_image(input_image, max_size)
            newHeight, newWidth = input_image.shape[:2]
            upsize_x_factor = Width / newWidth
            upsize_y_factor = Height / newHeight

        g.logger.debug(
            f"|---------- D-lib Face Detection and Recognition "
            f"{f'(original dimensions {Width}*{Height}) ' if newHeight else ''}"
            f"{f'(resized input image: {newWidth}*{newHeight}) ----------|' if newHeight else f'(input image: {Width}*{Height}) ----------|'}"
        )
        labels = []
        classes = []
        conf = []

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = input_image[:, :, ::-1]
        # rgb_image = image
        t = Timer()
        # Find all the faces and face encodings in the target image
        # g.logger.debug(self.options)
        if self.options.get('auto_lock', True):
            self.acquire_lock()

        face_locations = face_rec_libs.face_locations(
            rgb_image,
            model=self.face_model,
            number_of_times_to_upsample=self.upsample_times)

        diff_time = t.stop_and_get_ms()
        g.logger.debug(f"perf:face:{self.processor}: finding face locations took {diff_time}")

        t = Timer()
        face_encodings = face_rec_libs.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters)

        if self.options.get('auto_lock', True):
            self.release_lock()

        diff_time = t.stop_and_get_ms()
        g.logger.debug(
            f"perf:face:{self.processor}: computing face recognition distances took {diff_time}")

        if not len(face_encodings):
            return [], [], [], []

        # Use the KNN model to find the best matches for the test face

        g.logger.debug(3, "{lp} comparing detected faces to known faces...")

        t = Timer()
        if self.knn:
            # g.logger.debug(5, 'FACE ENCODINGS={}'.format(face_encodings))
            closest_distances = self.knn.kneighbors(face_encodings, n_neighbors=1)
            g.logger.debug(5, f"{lp} closest knn match indexes (smaller is better): {closest_distances}")
            are_matches = [
                closest_distances[0][i][0] <= float(self.options.get('face_recog_dist_threshold', 0.6))
                for i in range(len(face_locations))
            ]
            prediction_labels = self.knn.predict(face_encodings)
            g.logger.debug(5, f"{lp} KNN predictions: {prediction_labels} - are_matches: {are_matches}")

        else:
            g.logger.debug("{lp} no faces to match, creating empty set...")
            # There were no faces to compare
            # create a set of non matches for each face found
            are_matches = [False] * len(face_locations)
            prediction_labels = [''] * len(face_locations)

        diff_time = t.stop_and_get_ms()
        g.logger.debug(
            f"perf:face:{self.processor}: matching detected faces to known faces took {diff_time}")
        matched_face_names = []
        matched_face_rects = []
        if downscaled:
            g.logger.debug(2, f"{lp} scaling image back up to {Width}")
            input_image = old_image
            new_face_locations = []
            for loc in face_locations:
                a, b, c, d = loc
                a = round(a * upsize_y_factor)
                b = round(b * upsize_x_factor)
                c = round(c * upsize_y_factor)
                d = round(d * upsize_x_factor)
                new_face_locations.append((a, b, c, d))
            face_locations = new_face_locations

        for pred, loc, rec in zip(prediction_labels,
                                  face_locations, are_matches):
            label = pred if rec else self.options.get('unknown_face_name', 'unknown')
            if not rec and str2bool(self.options.get('save_unknown_faces')):
                h, w, c = input_image.shape
                x1 = max(loc[3] - int(self.options.get('save_unknown_faces_leeway_pixels', 0)), 0)
                y1 = max(loc[0] - int(self.options.get('save_unknown_faces_leeway_pixels', 0)), 0)
                x2 = min(loc[1] + int(self.options.get('save_unknown_faces_leeway_pixels', 0)), w)
                y2 = min(loc[2] + int(self.options.get('save_unknown_faces_leeway_pixels', 0)), h)
                # print (image)
                crop_img = input_image[y1:y2, x1:x2]
                # crop_img = image
                timestr = time.strftime("%b%d-%Hh%Mm%Ss-")
                unf = f"{self.options.get('unknown_images_path')}/{timestr}{id_generator()}.jpg"
                g.logger.info(
                    f"{lp} saving cropped '{self.options.get('unknown_face_name', 'unknown')}' face "
                    f"at [{x1},{y1},{x2},{y2} - includes leeway of "
                    f"{self.options.get('save_unknown_faces_leeway_pixels')}px] to {unf}")
                # cv2.imwrite wont throw an exception it outputs a WARN to console
                cv2.imwrite(unf, crop_img)

            matched_face_rects.append([loc[3], loc[0], loc[1], loc[2]])
            matched_face_names.append(label)
            # matched_face_names.append('face:{}'.format(label))
            conf.append(1)
        g.logger.debug(f"perf:{lp} total dlib sequence took {detect_start_timer.stop_and_get_ms()} ms")
        g.logger.debug(3, f'{lp} Returning -> {matched_face_rects}, {matched_face_names}, {conf}')
        return matched_face_rects, matched_face_names, conf, ['face_dlib'] * len(matched_face_names)
