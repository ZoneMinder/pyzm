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
        self.current_frame_to_read = 1
        self.options = options
        self.api = None
        self.more_images_to_read = True
        self.frames_processed = 0

        if options.get('logger'):
            self.logger = options.get('logger')
        elif options.get('api'):
            self.logger = options.get('api').get_logger() 

        self.start_frame = int(options.get('start_frame',1))
        self.frame_skip = int(options.get('frame_skip', 1))
        self.max_frames = int(options.get('max_frames', 0))

        if self.stream.isnumeric(): 
        # assume it is an eid
            if not self.options.get('api'):
                self.logger.Error('API object not provided, cannot download event {}'.format(self.stream))
                raise ValueError('API object not provided, cannot download event {}'.format(self.stream))
            
            self.api = self.options.get('api')
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
                self.current_frame_to_read = self.start_frame
                self.stream = '{}/index.php?view=image&width={}&eid={}'.format(self.api.get_portalbase(), 800, eid)
                if self.api.get_auth() != '':
                    self.stream += '&{}'.format(self.api.get_auth())

                self.logger.Debug(1,'Using URL {} for stream'.format(stream))
                self.type = 'image'
                
        
        if (self.type=='video'):
            self.logger.Debug (1, 'Starting video stream {}'.format(stream))
            self.fvs = FileVideoStream(self.stream).start()
            time.sleep(1)
            self.logger.Debug (1, 'Skipping to frame {}'.format(self.start_frame))
            while self.fvs.more() and self.current_frame_to_read < self.start_frame:
                self.fvs.read()
                self.current_frame_to_read += 1

        else:
            self.logger.Debug (1,'No need to start streams, we are picking images from {}'.format(self.stream))


    def get_last_read_frame(self):
        if self.type == 'image':
            return self.current_frame_to_read - self.frame_skip
        else:
            return self.current_frame_to_read - 1

    def more(self):
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
                if (self.current_frame_to_read - 1 ) % self.frame_skip:
                    self.current_frame_to_read += 1
                    continue
                else:
                    self.frames_processed += 1
                    self.logger.Debug(1, 'Processed video frame: {}'.format(self.current_frame_to_read))
                    self.current_frame_to_read += 1
                    break
            frame = imutils.resize(frame,width=800)
            return frame
        else:
            url = '{}&fid={}'.format(self.stream,self.current_frame_to_read)
            #self.logger.Debug (1, 'Reading {}'.format(url))

            response = requests.get(url)
            self.current_frame_to_read +=self.frame_skip
            self.frames_processed +=1 

            if response.status_code == 200:
                img = np.asarray(bytearray(response.content), dtype='uint8')
                img = cv2.imdecode (img, cv2.IMREAD_COLOR)
                return img
            else:
                self.more_images_to_read = False
                return None 
