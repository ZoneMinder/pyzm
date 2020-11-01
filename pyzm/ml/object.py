import numpy as np

import sys
import cv2
import time
import datetime
import re
from pyzm.helpers.Base import Base
from pyzm.helpers.Media import MediaStream

import time
import requests

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
            api (object): instance of the API if the stream need to route via ZM
            options (dict, optional): Various options that control the detection process. Defaults to {}:
            - 'download': boolean # if True, will download video before analysis
            - 'start_frame' int # Which frame to start analysis. Default 1.
            - 'frame_skip': int # Number of frames to skip in video (example, 3 means process every 3rd frame)
            - 'max_frames' : int # Total number of frames to process before stopping
            - 'pattern': string # regexp for objects that will be matched. 'strategy' key below will be applied
                         to only objects that match this pattern
            - 'strategy': string # various conditions to stop matching as below
                'first' # stop at first match (Default)
                'most' # match the frame that has the highest number of detected objects
                'most_unique' # match the frame that has the highest number of unique detected objects
    
            Returns:
                boxes (array): list of bounding boxes for matched frame
                labels (array): list of labels for matched frame
                confidences (array): list of confidences for matched frame
                id (int): frame id of matched frame
                img (cv2 image): image grab of matched frame
                all_matches (array of objects): list of boxes,labels,confidences of all frames matched

            Note:
            The same frames are not retrieved depending on whether you set
            ``download`` to ``True`` or ``False``. When set to ``True``, we use
            OpenCV's frame reading logic and when ``False`` we use ZoneMinder's image.php function
            which uses time based approximation. Therefore, the retrieve different frame offsets, but I assume
            they should be reasonably close.
            
        """

        

              
        self.model.acquire_lock()
        
        match_strategy = options.get('strategy', 'first')
        match_pattern = re.compile(options.get('pattern', '.*'))

        all_matches = []
        matched_b = []
        matched_l = []
        matched_c = []
        matched_frame_id = None
        matched_frame_img = None

        start = datetime.datetime.now()
        media = MediaStream(stream,'video', options )
        while media.more():
            frame = media.read()
            if frame is None:
                self.logger.Debug(1,'Ran out of frames to read')
                break
                    
            b,l,c  =self.model.detect(image=frame, only_detect=True)
            #print ('LABELS {} BOXES {}'.format(l,b))
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
            all_matches.append (
                {
                    'frame_id': media.get_last_read_frame(),
                    'boxes': b,
                    'labels': l,
                    'confidences': c
                }
            )
            if  ((match_strategy == 'first') 
                or ((match_strategy == 'most') and (len(l) > len(matched_l))) 
                or ((match_strategy == 'most_unique') and (len(set(l)) > len(set(matched_l))))):
                matched_b = b
                matched_c = c
                matched_l = l
                matched_frame_id = media.get_last_read_frame()
                matched_frame_img = frame
            
            if (match_strategy=='first'):
                break
            
        # release resources
        diff_time = (datetime.datetime.now() - start).microseconds / 1000
        self.logger.Debug(
            1,'Coral TPU detection took: {} milliseconds to process {}'.format(diff_time, stream))
        media.stop()
        self.model.release_lock()
        return matched_b, matched_l, matched_c, matched_frame_id, matched_frame_img, all_matches


        
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
