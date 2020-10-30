import numpy as np

import sys
import cv2
import time
import datetime
import re
from pyzm.helpers.Base import Base
import imutils

from imutils.video import FileVideoStream
from imutils.video import FPS
import time

# Class to handle Yolo based detection




class Object(Base):

    
    def __init__(self, options={}, logger=None):

        Base.__init__(self,logger)
        self.model = None
        self.options = options

        if self.options.get('object_framework') == 'opencv':
            import pyzm.ml.yolo as yolo
            self.model =  yolo.Yolo(options=options, logger=logger)
            

        elif self.options.get('object_framework') == 'coral_edgetpu':
            import pyzm.ml.coral_edgetpu as tpu
            self.model = tpu.Tpu(options=options, logger=logger)

        else:
            raise ValueError ('Invalid object_framework:{}'.format(self.options.get('object_framework')))

    def get_model(self):
            return self.model

    def get_classes(self):
            return self.model.get_classes()


    def detect_stream(self, stream, options={}):
        """Implements detection on a video stream

        Args:
            stream (string): location of media (file, url or event ID)
            options (dict, optional): Various options that control the detection process. Defaults to {}:
            - 'start_frame' int # Which frame to start analysis. Default 1.
            - 'frame_skip': int # Number of frames to skip in video (example, 3 means process every 3rd frame)
            - 'max_frames' : int # Total number of frames to process before stopping
            - 'pattern': string # regexp for objects that will be matched. 'strategy' key below will be applied
                         to only objects that match this pattern
            - 'strategy': string # various conditions to stop matching as below
                'first' # stop at first match (Default)
                'most' # match the frame that has the highest number of detected objects
                'most_unique' # match the frame that has the highest number of unique detected objects
            
        """

        self.model.acquire_lock()
        fvs = FileVideoStream(stream).start()    
        time.sleep(1.0)
        fps = FPS().start()

        start_frame = int(options.get('start_frame',1))
        frame_skip = int(options.get('frame_skip', 1))
        max_frames = int(options.get('max_frames', 0))
        match_strategy = options.get('strategy', 'first')
        match_pattern = re.compile(options.get('pattern', '.*'))

        
        frames_read = 0
        frames_processed = 0

        matched_b = []
        matched_l = []
        matched_c = []
        matched_frame_id = None
        matched_frame_img = None

        start = datetime.datetime.now()
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                self.logger.Debug(1,'Ran out of frames to read at {}'.format(frames_read))
                break
            frames_read +=1
            fps.update()
            if frames_read < start_frame:
                continue
            if frames_read % frame_skip:
               continue
            if max_frames and frames_processed >= max_frames:
                self.logger.Debug(1, 'Bailing as {} frames processed'.format(frames_processed))
                break
            
            # The API expects non-raw images, so lets convert to jpg
            # ret, jpeg = cv2.imencode('.jpg', frame)
            frame = imutils.resize(frame,width=800)
          
            b,l,c  =self.model.detect(image=frame, only_detect=True)
            frames_processed +=1
            f_b = []
            f_l = []
            f_c = []

            match = list(filter(match_pattern.match, l))
            for idx, label in enumerate(l):
                if label not in match:
                    continue
                f_b.append(b[idx])
                f_l.append(label)
                f_c.append(c[idx])

            l = f_l
            b = f_b
            c = f_c

            if not len(l):
                continue
            #print ('Frame:{}, labels:{}'.format(frames_processed, l))
            if match_strategy == 'first':
                matched_b = b
                matched_c = c
                matched_l = l
                matched_frame_id = frames_read
                matched_frame_img = frame
                break
            elif match_strategy == 'most':
                if (len(l) > len(matched_l)):
                    matched_b = b
                    matched_c = c
                    matched_l = l
                    matched_frame_id = frames_read
                    matched_frame_img = frame
            elif match_strategy == 'most_unique':
                 if (len(set(l)) > len(set(matched_l))):
                    matched_b = b
                    matched_c = c
                    matched_l = l
                    matched_frame_id = frames_read
                    matched_frame_img = frame

        # release resources
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'Coral TPU detection took: {} milliseconds to process {}'.format(diff_time, stream))
        fvs.stop()
        self.model.release_lock()
        return matched_b, matched_l, matched_c, matched_frame_id, matched_frame_img


        
    def detect(self,image=None):
        h,w = image.shape[:2]
        b,l,c = self.model.detect(image)
        self.logger.Debug (2,'core model detection over, got {} objects. Now filtering'.format(len(b)))
        # Apply various object filtering rules
        max_object_area = 0
        if self.options.get('max_detection_size'):
                self.logger.Debug(3,'Max object size found to be: {}'.format(self.options.get('max_detection_size')))
                # Let's make sure its the right size
                m = re.match('(\d*\.?\d*)(px|%)?$', self.options.get('max_detection_size'),
                            re.IGNORECASE)
                if m:
                    max_object_area = float(m.group(1))
                    if m.group(2) == '%':
                        max_object_area = float(m.group(1))/100.0*(h * w)
                        self.logger.Debug (2,'Converted {}% to {}'.format(m.group(1), max_object_area))
                else:
                    self.logger.Error('max_object_area misformatted: {} - ignoring'.format(
                        self.options.get('max_object_area')))

        boxes=[]
        labels=[]
        confidences=[]

        for idx,box in enumerate(b):
            (sX,sY,eX,eY) = box
            if max_object_area:
                object_area = abs((eX-sX)*(eY-sY))
                if (object_area > max_object_area):
                    self.logger.Debug (1,'Ignoring object:{}, as it\'s area: {}px exceeds max_object_area of {}px'.format(l[idx], object_area, max_object_area))
                    continue
            if c[idx] >= self.options.get('object_min_confidence'):
                boxes.append([sX,sY,eX,eY])
                labels.append(l[idx])
                confidences.append(c[idx])
            else:
                self.logger.Debug (1,'Ignoring {} {} as conf. level {} is lower than {}'.format(l[idx],box,c[idx],self.options.get('object_min_confidence')))
       
        self.logger.Debug (2,'Returning filtered list of {} objects.'.format(len(boxes)))
        return boxes,labels,confidences
