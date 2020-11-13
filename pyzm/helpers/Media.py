"""
Media
======
Generic media class that can read frames from either
a media file or via ZM APIs/index.php

"""

from pyzm.helpers.Base import Base
import requests
import cv2
import numpy as np
import imutils
from imutils.video import FileVideoStream
import time

class MediaStream(Base):
    def __init__(self, stream=None, type='video', options={}):
        self.stream = stream
        self.type = type
        self.fvs = None
        self.next_frameid_to_read = 1
        self.last_frameid_read = 0
        self.options = options
        self.more_images_to_read = True
        self.frames_processed = 0
        self.frames_read = 0
        self.api = self.options.get('api')
        self.frame_set = []
        self.next_frame_set_index = 0
        

        if self.options.get('logger'):
            self.logger = options.get('logger')
        elif self.options.get('api'):
            self.logger = options.get('api').get_logger() 

       
        self.start_frame = int(options.get('start_frame',1))
        self.frame_skip = int(options.get('frame_skip', 1))
        self.max_frames = int(options.get('max_frames', 0))

        
        if self.stream.isnumeric(): 
        # assume it is an eid, in which case we 
        # need to access it via ZM API
            if not self.options.get('api'):
                self.logger.Error('API object not provided, cannot download event {}'.format(self.stream))
                raise ValueError('API object not provided, cannot download event {}'.format(self.stream))
            
            # we can either download the event, or access it via ZM
            # indirection 
            if self.options.get('download'):
                self.logger.Debug(1,'Downloading event:{}'.format(self.stream))
                es = self.api.events({'event_id': int(self.stream)})
                if not len(es.list()):
                    self.logger.Error('No such event {} found with API'.format(self.stream))
                    return
                e = es.list()[0]
                self.stream = e.download_video(dir='/tmp')
                
                self.type = 'video'
            else:
                eid = self.stream
                self.next_frameid_to_read = int(self.start_frame)
                self.stream = '{}/index.php?view=image&width={}&eid={}'.format(self.api.get_portalbase(), self.options.get('resize', 800), eid)
                if self.api.get_auth() != '':
                    self.stream += '&{}'.format(self.api.get_auth())

                self.logger.Debug(1,'Using URL {} for stream'.format(stream))
                self.type = 'image'
        
        
        if self.options.get('frame_set'):
            self.frame_set = self.options.get('frame_set').split(',')
            if 'alarm' in self.frame_set or 'snapshot' in self.frame_set:
                if not self.api or self.type == 'video':
                    # if we are not using ZM indirection, we cannot use 'alarm' 'snapshot' etc.
                
                    self.logger.Error ('You are using frame_types that require ZM indirection')
                    raise ValueError ('You are using frame_types that require ZM indirection')
            self.max_frames = len (self.frame_set)
            self.logger.Debug (2, 'We will only process frames: {}'.format(self.frame_set))
            self.next_frame_set_index = 0
            self.start_frame = self.frame_set[self.next_frame_set_index]
    
        if (self.type=='video'):
            self.logger.Debug (1, 'Starting video stream {}'.format(stream))
            self.fvs = FileVideoStream(self.stream).start()
            time.sleep(1)
            self.logger.Debug (1, 'First load - skipping to frame {}'.format(self.start_frame))
            if self.frame_set:
                while self.fvs.more() and self.frames_read < int(self.frame_set[self.next_frame_set_index])-1:
                    self.fvs.read() 
                    self.frames_read +=1

            else:
                while self.fvs.more() and self.next_frameid_to_read < int(self.start_frame):
                    self.fvs.read()
                    self.next_frameid_to_read += 1

        else:
            self.logger.Debug (1,'No need to start streams, we are picking images from {}'.format(self.stream))


    def get_last_read_frame(self):
        return self.last_frameid_read

    def more(self):
        if (self.frame_set):
            return self.next_frame_set_index < len(self.frame_set)

        if (self.frames_processed >= self.max_frames):
            self.logger.Debug(1, 'Bailing as we have read {} frames'.format(self.max_frames))
            return False
        else:
            if (self.type == 'video'):
                return self.fvs.more()
            else:
                return self.more_images_to_read

    def stop(self):
        if (self.type == 'video'):
            self.fvs.stop()

    def read(self):
        
        if (self.type == 'video'):
            while True:
                frame = self.fvs.read()
                self.frames_read +=1

                if frame is None:
                    self.logger.Error ('Error reading frames')
                    return

                if self.frame_set and (self.frames_read != int(self.frame_set[self.next_frame_set_index])):
                    self.logger.Debug (4,'Ignoring frame {}'.format(self.frames_read))
                    continue
                else:
                    self.logger.Debug (4,'MATCHED frame {}'.format(self.frames_read))                    

                    # At this stage weare at the frame to read
                if self.frame_set:
                    self.last_frameid_read = self.frame_set[self.next_frame_set_index]
                    self.next_frame_set_index += 1
                    if self.next_frame_set_index < len (self.frame_set):
                        self.logger.Debug (4,"Now moving to frame: {}".format(self.frame_set[self.next_frame_set_index])) 

                else:
                    self.last_frameid_read = self.next_frameid_to_read
                    self.next_frameid_to_read +=1
                    if (self.last_frameid_read - 1 ) % self.frame_skip:
                        self.logger.Debug(5,'Skipping frame {}'.format(self.last_frameid_read))
                        continue    
                    
                self.logger.Debug(2, 'Processing frame:{}'.format(self.last_frameid_read))
                self.frames_processed += 1
                break
            frame = imutils.resize(frame,width=self.options.get('resize', 800))
            return frame
        
        else: # image
            if self.frame_set:
                url = '{}&fid={}'.format(self.stream,self.frame_set[self.next_frame_set_index])
            else:
                url = '{}&fid={}'.format(self.stream,self.next_frameid_to_read)
            
            self.logger.Debug (2, 'Reading {}'.format(url))

            response = requests.get(url)
            if self.frame_set:
                self.last_frameid_read = self.frame_set[self.next_frame_set_index]
                self.next_frame_set_index += 1
            else:
                self.last_frameid_read = self.next_frameid_to_read
                self.next_frameid_to_read +=self.frame_skip
            self.frames_processed +=1 

            if response.status_code == 200:
                try:
                    img = np.asarray(bytearray(response.content), dtype='uint8')
                    img = cv2.imdecode (img, cv2.IMREAD_COLOR)
                    return img
                except Exception as e:
                    self.logger.Error ('Could not retrieve url {}: {}'.format(url,e))
                    return None
            else:
                self.more_images_to_read = False
                self.next_frameid_to_read = 0
                return None 
