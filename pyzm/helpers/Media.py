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
import os
import random
import pyzm.helpers.globals as g


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
        self.orig_h_w = None
        self.resized_h_w = None        
        self.default_resize = 800
        self.is_deletable = False
        self.session = requests.Session()
        self.eid=None
     
        self.debug_filename = None 



        if options.get('delay'):
            g.logger.Debug(1, 'Waiting for {} seconds'.format(options.get('delay')))
            time.sleep(int(options.get('delay')))


        if options.get('disable_ssl_cert_check', True):
            self.session.verify = False
            g.logger.Debug (2, 'Media get SSL certificate check has been disbled')
            from urllib3.exceptions import InsecureRequestWarning
            requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

        _,ext= os.path.splitext(self.stream)
        if ext.lower() in ['.jpg', '.png', '.jpeg']:
            g.logger.Debug(1, '{} is a file'.format(self.stream))
            self.type = 'file'
            return
       
        self.start_frame = int(options.get('start_frame',1))
        self.frame_skip = int(options.get('frame_skip', 1))
        self.max_frames = int(options.get('max_frames', 0))
        self.contig_frames_before_error = int(options.get('contig_frames_before_error',5))
        self.frames_before_error = 0
        
        
        if self.stream.isnumeric(): 
        # assume it is an eid, in which case we 
        # need to access it via ZM API
            if not self.options.get('api'):
                g.logger.Error('API object not provided, cannot download event {}'.format(self.stream))
                raise ValueError('API object not provided, cannot download event {}'.format(self.stream))
            
            # we can either download the event, or access it via ZM
            # indirection 
            self.debug_filename = '{}-image'.format(self.stream)
            if self.options.get('download'):
                g.logger.Debug(1,'Downloading event:{}'.format(self.stream))
                es = self.api.events({'event_id': int(self.stream)})
                if not len(es.list()):
                    g.logger.Error('No such event {} found with API'.format(self.stream))
                    return
                e = es.list()[0]
                self.stream = e.download_video(dir=self.options.get('download_dir', '/tmp/'))
                self.is_deletable = True
                self.type = 'video'
            else:
                eid = self.stream
                self.eid = eid
                self.next_frameid_to_read = int(self.start_frame)
                self.stream = '{}/index.php?view=image&eid={}'.format(self.api.get_portalbase(),  eid)
                #if self.options.get('resize') != 'no':
                #    self.stream += '&width={}'.format(self.options.get('resize', self.default_resize))

                #if self.api.get_auth() != '':
                #    self.stream += '&{}'.format(self.api.get_auth())

                g.logger.Debug(2,'Using URL {} for stream'.format(stream))
                self.type = 'image'
        
        
        if self.options.get('frame_set'):
            self.frame_set = self.options.get('frame_set').split(',')
            if 'alarm' in self.frame_set or 'snapshot' in self.frame_set:
                if not self.api or self.type == 'video':
                    # if we are not using ZM indirection, we cannot use 'alarm' 'snapshot' etc.
                
                    g.logger.Error ('You are using frame_types that require ZM indirection')
                    raise ValueError ('You are using frame_types that require ZM indirection')
            self.max_frames = len (self.frame_set)
            g.logger.Debug (2, 'We will only process frames: {}'.format(self.frame_set))
            self.next_frame_set_index = 0
            self.start_frame = self.frame_set[self.next_frame_set_index]
    
        if (self.type=='video'):
            g.logger.Debug (1, 'Starting video stream {}'.format(stream))
            f,_=os.path.splitext(os.path.basename(self.stream))
            self.debug_filename = '{}-image'.format(f)

            self.fvs = FileVideoStream(self.stream).start()
            time.sleep(1)
            g.logger.Debug (2, 'First load - skipping to frame {}'.format(self.start_frame))
            if self.frame_set:
                while self.fvs.more() and self.frames_read < int(self.frame_set[self.next_frame_set_index])-1:
                    self.fvs.read() 
                    self.frames_read +=1

            else:
                while self.fvs.more() and self.next_frameid_to_read < int(self.start_frame):
                    self.fvs.read()
                    self.next_frameid_to_read += 1

        else:
            g.logger.Debug (2,'No need to start streams, we are picking images from {}'.format(self.stream))

    def get_debug_filename(self):
        return self.debug_filename

    def image_dimensions(self):
        return {
            'original': self.orig_h_w,
            'resized': self.resized_h_w
        }

    def get_last_read_frame(self):
        return self.last_frameid_read

    def more(self):

        if self.type == 'file':
            return self.more_images_to_read

        if (self.frame_set):
            return self.next_frame_set_index < len(self.frame_set)

        if (self.frames_processed >= self.max_frames):
            g.logger.Debug(1, 'Bailing as we have read {} frames'.format(self.max_frames))
            return False
        else:
            if (self.type == 'video'):
                return self.fvs.more()
            else:
                return self.more_images_to_read

    def stop(self):
        if (self.type == 'video'):
            self.fvs.stop()
        if self.is_deletable and self.options.get('delete_after_analyze') == 'yes':
            try:
                os.remove(self.stream)
                g.logger.Debug(2,'Deleted {}'.format(self.stream))
            except Exception as e:
                g.logger.Error (f'Could not delete file(s):{e}')

    def read(self):
        if (self.type == 'file'):
            frame = cv2.imread(self.stream)
            self.last_frame_id_read = 1
            self.orig_h_w = frame.shape[:2]
            self.frames_processed += 1
            self.more_images_to_read = False
            if self.options.get('resize') != 'no':
                frame = imutils.resize(frame,width=self.options.get('resize', self.default_resize))
            self.resized_h_w = frame.shape[:2]  
            return frame


        if (self.type == 'video'):
            while True:
                frame = self.fvs.read()
                self.frames_read +=1

                if frame is None:
                    self.frames_before_error +=1
                    if (self.frames_before_error >= self.contig_frames_before_error):
                        g.logger.Error ('Error reading frames')
                        return
                    else:
                        g.logger.Debug (1, 'Error reading frame: {} of max {} contiguous errors'.format(self.frames_before_error, self.contig_frames_before_error))
                        continue

                self.frames_before_error = 0
                self.orig_h_w = frame.shape[:2]
                if self.frame_set and (self.frames_read != int(self.frame_set[self.next_frame_set_index])):
                    continue
                    # At this stage we are at the frame to read
                if self.frame_set:
                    self.last_frameid_read = self.frame_set[self.next_frame_set_index]
                    self.next_frame_set_index += 1
                    if self.next_frame_set_index < len (self.frame_set):
                        g.logger.Debug (4,"Now moving to frame: {}".format(self.frame_set[self.next_frame_set_index])) 

                else:
                    self.last_frameid_read = self.next_frameid_to_read
                    self.next_frameid_to_read +=1
                    if (self.last_frameid_read - 1 ) % self.frame_skip:
                        g.logger.Debug(5,'Skipping frame {}'.format(self.last_frameid_read))
                        continue    
                    
                g.logger.Debug(2, 'Processing frame:{}'.format(self.last_frameid_read))
                self.frames_processed += 1
                break
            #print ('******************************* RESIZE:{}'.format(self.options.get('resize')))
            if self.options.get('resize') != 'no':
                frame = imutils.resize(frame,width=self.options.get('resize', self.default_resize))
            self.resized_h_w = frame.shape[:2]

            if self.options.get('save_frames') and self.debug_filename:
                d = self.options.get('save_frames_dir','/tmp')
                fname = '{}/{}-{}.jpg'.format(d,self.debug_filename,self.last_frameid_read) 
                g.logger.Debug (4, 'Saving image to {}'.format(fname))
                cv2.imwrite(fname,frame)        

            return frame
        
        else: # image
            if self.frame_set:
                if self.next_frame_set_index >= len (self.frame_set):
                    g.logger.Debug (1,'Reached end of frame_set')
                    self.more_images_to_read = False
                    self.next_frameid_to_read = 0
                    return None 
                if self.frame_set[self.next_frame_set_index]=='snapshot' and self.api:
                    g.logger.Debug(4,'Trying to convert snapshot to a real frame id')
                    try:
                        eurl = '{}/events/{}.json'.format(self.api.get_apibase(),self.eid)
                        res = self.api._make_request(eurl)
                        fid = res.get('event',{}).get('Event',{}).get('MaxScoreFrameId')
                        g.logger.Debug(4,'At the point of analysis, Event:{} snapshot frame id was:{},so using it'.format(self.eid, fid))
                        self.frame_set[self.next_frame_set_index]= str(fid)
                    except Exception as e:
                        g.logger.Debug(4,' Failed retrieving snapshot frame ID:{}'.format(e))

                url = '{}&fid={}'.format(self.stream,self.frame_set[self.next_frame_set_index])

                # do we need snapshot deref?
            else:
                url = '{}&fid={}'.format(self.stream,self.next_frameid_to_read)
            
            g.logger.Debug (3, 'Reading {}'.format(url))
            response = None
            try:
                if self.api:
                    attempts = 1
                    max_attempts = self.options.get('max_attempts',1)
                    sleep_time = self.options.get('sleep_between_attempts',3)
                    while attempts <= max_attempts:
                        try:
                            response = self.api._make_request(url)
                        except Exception as e:
                            err_msg = '{}'.format(e)
                            if err_msg != 'BAD_IMAGE':
                                break
                            g.logger.Debug (2,'Failed attempt:{} of {}'.format(attempts, max_attempts))
                            attempts += 1
                            if sleep_time and attempts <=max_attempts:
                                g.logger.Debug (2,'Sleeping for {} seconds before retry'.format(sleep_time))
                                time.sleep(sleep_time)
                        else: # request worked, no need to retry
                            break
                else:
                    response = self.session.get(url)
            except Exception as e:
                if self.frame_set:
                    self.next_frame_set_index += 1
                    if self.next_frame_set_index < len(self.frame_set):
                        g.logger.Error('Error reading frame: {}, but moving to next frame_set'.format(url))
                        self.last_frameid_read = self.frame_set[self.next_frame_set_index]
                        return self.read()
                    else:
                        g.logger.Error('Error reading frame: {} and no more frames to read'.format(url))
                        self.more_images_to_read = False
                        self.next_frameid_to_read = 0
                        return None 
                        
            if self.frame_set:
                self.last_frameid_read = self.frame_set[self.next_frame_set_index]
                self.next_frame_set_index += 1
            else:
                self.last_frameid_read = self.next_frameid_to_read
                self.next_frameid_to_read +=self.frame_skip

            self.frames_processed +=1 
            if response and response.status_code == 200: # and random.randint(0,5)==1:
                self.frames_before_error = 0
                try:
                    img = np.asarray(bytearray(response.content), dtype='uint8')
                    img = cv2.imdecode (img, cv2.IMREAD_COLOR)
                    self.orig_h_w =  img.shape[:2]
                    #print ('******************************* RESIZE:{}'.format(self.options.get('resize')))
                    if self.options.get('resize') != 'no':
                        img = imutils.resize(img,width=self.options.get('resize', self.default_resize))
                    self.resized_h_w = img.shape[:2]

                    if self.options.get('save_frames') and self.debug_filename:
                        d = self.options.get('save_frames_dir','/tmp')
                        fname = '{}/{}-{}.jpg'.format(d,self.debug_filename,self.last_frameid_read) 
                        g.logger.Debug (4, 'Saving image to {}'.format(fname))
                        cv2.imwrite(fname,img)        

                    return img
                except Exception as e:
                    g.logger.Error ('Could not retrieve url {}: {}'.format(url,e))
                    return None
          
            else: # response code not 200
                self.frames_before_error +=1
                if (self.frames_before_error >= self.contig_frames_before_error):
                    g.logger.Error ('Error reading frames')
                    return None
                else:
                    g.logger.Debug (1, 'Error reading frame: {} of max {} contiguous errors'.format(self.frames_before_error, self.contig_frames_before_error))
                    return self.read()
                self.more_images_to_read = False
                self.next_frameid_to_read = 0
                return None 
