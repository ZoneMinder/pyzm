import numpy as np
import cv2
import requests
import os
import imutils
import json
import base64
import subprocess
import uuid
from pyzm.helpers.Base import Base

class AlprBase(Base):
    def __init__(self, logger=None,options={}, tempdir='/tmp'):
        super().__init__(logger)
        if not options.get('alpr_key'):
            self.logger.Debug (1,'No API key specified, hopefully you do not need one')
        self.apikey = options.get('alpr_key')
        #print (self.apikey)
        self.tempdir = tempdir
        self.url = options.get('alpr_url')
        self.options = options
        

    def setkey(self, key=None):
        self.apikey = key
        self.logger.Debug(2,'Key changed')

    def stats(self):
        self.logger.Debug(1,'stats not implemented in base class')

    def detect(self, object):
        self.logger.Debug(1,'detect not implemented in base class')

    def prepare(self, object):
        if not isinstance(object, str):
            self.logger.Debug(
                1,'Supplied object is not a file, assuming blob and creating file'
            )
            if self.options.get('resize') and self.options.get('resize') != 'no':
                self.logger.Debug(2, 'resizing image blob to {}'.format(self.options.get('resize')) )
                obj_new = imutils.resize(object, width=min(int(self.options.get('resize')),
                                               object.shape[1]))
                object = obj_new
            # use png so there is no loss
            self.filename = self.tempdir + '/'+str(uuid.uuid4())+'-alpr.png'
            cv2.imwrite(self.filename, object)
            self.remove_temp = True
        else:
            # If it is a file and zm_detect sent it, it would already be resized
            # If it is a file and zm_detect did not send it, no need to adjust scales
            # as there won't be a yolo/alpr size mismatch
            self.logger.Debug(1,f'supplied object is a file {object}')
            self.filename = object
           
            self.remove_temp = False
  

    def getscale(self):
        if self.options.get('resize') and self.options.get('resize') != 'no':
            img = cv2.imread(self.filename)
            img_new = imutils.resize(img,
                                     width=min(int(self.options.get('resize')),
                                               img.shape[1]))
            oldh, oldw, _ = img.shape
            newh, neww, _ = img_new.shape
            rescale = True
            xfactor = neww / oldw
            yfactor = newh / oldh
            img = None
            img_new = None
            self.logger.Debug(
                2,'ALPR will use {}x{} but Object uses {}x{} so ALPR boxes will be scaled {}x and {}y'
                .format(oldw, oldh, neww, newh, xfactor, yfactor))
        else:
            xfactor = 1
            yfactor = 1
        return (xfactor, yfactor)


class Alpr(AlprBase):
    def __init__(self, options={},logger=None, tempdir='/tmp'):
        """Wrapper class for all ALPR objects

        Args:
            options (dict, optional): Config options. Defaults to {}.
            tempdir (str, optional): Path to store image for analysis. Defaults to '/tmp'.
        """        
        AlprBase.__init__(self, options=options,logger=logger, tempdir=tempdir)
        self.alpr_obj = None

        if self.options.get('alpr_service') == 'plate_recognizer':   
            self.alpr_obj = PlateRecognizer(options=self.options, logger=logger)
        elif self.options.get('alpr_service') == 'open_alpr':   
            self.alpr_obj = OpenAlpr(options=self.options, logger=logger)
        elif self.options.get('alpr_service') == 'open_alpr_cmdline':   
            self.alpr_obj = OpenAlprCmdLine(options=self.options, logger=logger)
                  
        else:
            raise ValueError('ALPR service "{}" not known'.format(self.options.get('alpr_service')))

    def detect(self, object):
        return self.alpr_obj.detect(object)

    def stats(self):
        return self.alpr_obj.stats()



class PlateRecognizer(AlprBase):

    def __init__(self, options={}, logger=None, tempdir='/tmp'):
        """Wrapper class for platerecognizer.com

        Args:
            options (dict, optional): Config options. Defaults to {}.
            tempdir (str, optional): Path to store image for analysis. Defaults to '/tmp'.
        """        
        AlprBase.__init__(self, options=options,  tempdir=tempdir, logger=logger)
        
        url=self.options.get('alpr_url')
        apikey=self.options.get('alpr_key')
        
        
        if not url:
            self.url = 'https://api.platerecognizer.com/v1'

        self.logger.Debug(
            1,'PlateRecognizer ALPR initialized with url: {}'.
            format(self.url))
        

    def stats(self):
        """Returns API statistics

        Returns:
            HTTP Response: HTTP response of statistics API
        """        
        if self.options.get('alpr_api_type') != 'cloud':
            self.logger.Debug (1,'local SDK does not provide stats')
            return {}
        try:
            if self.apikey:
                 headers={'Authorization': 'Token ' + self.apikey}
            else:
                headers={}
            response = requests.get(
                self.url + '/statistics/',
               headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            response = {'error': str(e)}
        else:
            response = response.json()
        return response

    def detect(self, object):
        """Detects license plate using platerecognizer

        Args:
            object (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """        
        bbox = []
        labels = []
        confs = []
        self.prepare(object)
        if self.options.get('platerec_stats') == 'yes':
            self.logger.Debug(1,'Plate Recognizer API usage stats: {}'.format(
                json.dumps(self.stats())))
        with open(self.filename, 'rb') as fp:
            try:
                platerec_url = self.url
                if self.options.get('alpr_api_type') == 'cloud':
                    platerec_url += '/plate-reader'
                payload = self.options.get('platerec_regions')
                response = requests.post(
                   platerec_url,
                    #self.url ,
                    files=dict(upload=fp),
                    data=payload,
                    headers={'Authorization': 'Token ' + self.apikey})
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                response = {
                    'error':
                    f'Plate recognizer rejected the upload with: {e}.',
                    'results': []
                }
                self.logger.Error(
                    f'Plate recognizer rejected the upload with {e}'
                )
            else:
                response = response.json()
                self.logger.Debug(2,'ALPR JSON: {}'.format(response))

        #(xfactor, yfactor) = self.getscale()

        if self.remove_temp:
            os.remove(self.filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                dscore = plates['dscore']
                score = plates['score']
                if dscore >= self.options.get('platerec_min_dscore') and score >= self.options.get(
                        'platerec_min_score'):
                    x1 = round(int(plates['box']['xmin']))
                    y1 = round(int(plates['box']['ymin']))
                    x2 = round(int(plates['box']['xmax']))
                    y2 = round(int(plates['box']['ymax']))
                    labels.append(label)
                    bbox.append([x1, y1, x2, y2])
                    confs.append(plates['score'])
                else:
                    self.logger.Debug(
                        1,'ALPR: discarding plate:{} because its dscore:{}/score:{} are not in range of configured dscore:{} score:{}'
                        .format(label, dscore, score, self.options.get('platerec_min_dscore'),
                                self.options.get('platerec_min_score')))

        self.logger.Debug (2,'Exiting ALPR with labels:{}'.format(labels))
        return (bbox, labels, confs)


class OpenAlpr(AlprBase):
    def __init__(self, options={},logger=None, tempdir='/tmp'):
        """Wrapper class for Open ALPR service

        Args:
            options (dict, optional): Various ALPR options. Defaults to {}.
            tempdir (str, optional): Temporary path to analyze image. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options, logger=logger, tempdir=tempdir)
        if not url:
            self.url = 'https://api.openalpr.com/v2/recognize'

        self.logger.Debug(
            1,'Open ALPR initialized with url: {}'.
            format(self.url))
        

    def detect(self, object):
        """Detection using OpenALPR

        Args:
            object (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """        
        bbox = []
        labels = []
        confs = []

        self.prepare(object)
     
        with open(self.filename, 'rb') as fp:
            try:
               
                params = ''
                if self.options.get('openalpr_country'):
                    params = params + '&country=' + self.options.get('oenalpr_country')
                if self.options.get('openalpr_state'):
                    params = params + '&state=' + self.options.get('openalpr_state')
                if self.options.get('openalpr_recognize_vehicle'):
                    params = params + '&recognize_vehicle=' + \
                        str(self.options.get('openalpr_recognize_vehicle'))

                rurl = '{}?secret_key={}{}'.format(self.url, self.apikey,
                                                   params)
                self.logger.Debug(1,'Trying OpenALPR with url:' + rurl)
                response = requests.post(rurl, files={'image': fp})
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                response = {
                    'error':
                    f'Open ALPR rejected the upload with {e}',
                    'results': []
                }
                self.logger.Debug(
                    1,f'Open APR rejected the upload with {e}'
                )
            else:
                response = response.json()
                self.logger.Debug(1,'OpenALPR JSON: {}'.format(response))

        #(xfactor, yfactor) = self.getscale()

        rescale = False

        if self.remove_temp:
            os.remove(filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                conf = float(plates['confidence']) / 100
                if conf < options.get('openalpr_min_confidence'):
                    self.logger.Debug(
                        1,'OpenALPR: discarding plate: {} because detected confidence {} is less than configured min confidence: {}'
                        .format(label, conf, self.options.get('openalpr_min_confidence')))
                    continue

                if plates.get(
                        'vehicle'):  # won't exist if recognize_vehicle is off
                    veh = plates.get('vehicle')
                    for attribute in ['color', 'make', 'make_model', 'year']:
                        if veh[attribute]:
                            label = label + ',' + veh[attribute][0]['name']

                x1 = round(int(plates['coordinates'][0]['x']))
                y1 = round(int(plates['coordinates'][0]['y']))
                x2 = round(int(plates['coordinates'][2]['x']))
                y2 = round(int(plates['coordinates'][2]['y']))
                labels.append(label)
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return (bbox, labels, confs)

class OpenAlprCmdLine(AlprBase):
    def __init__(self, options={}, logger=None, tempdir='/tmp'):
        """Wrapper class for OpenALPR command line utility

        Args:
            cmd (string, optional): The cli command. Defaults to None.
            options (dict, optional): Various ALPR options. Defaults to {}.
            tempdir (str, optional): Temporary path to analyze image. Defaults to '/tmp'.
        """        
        AlprBase.__init__(self, options=options, logger=logger, tempdir=tempdir)
        
        cmd=self.options.get('openalpr_cmdline_binary')

                                                
        self.cmd = cmd + ' ' + self.options.get('openalpr_cmdline_params')
        if self.cmd.lower().find('-j') == -1:
            self.logger.Debug (2,'Adding -j to OpenALPR for json output')
            self.cmd = self.cmd + ' -j'
      

    def detect(self, object):
        """Detection using OpenALPR command line

        Args:
            object (image): image buffer

         Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """             
        bbox = []
        labels = []
        confs = []

        self.prepare(object)
        self.cmd = self.cmd + ' ' + self.filename
        self.logger.Debug (1,'OpenALPR CmdLine Executing: {}'.format(self.cmd))
        response = subprocess.check_output(self.cmd, shell=True)      
        self.logger.Debug (1,'OpenALPR CmdLine Response: {}'.format(response))
        try:
            response = json.loads(response)
        except ValueError as e:
            self.logger.Error ('Error parsing JSON from command line: {}'.format(e))
            response = {}

        #(xfactor, yfactor) = self.getscale()

        rescale = False

        if self.remove_temp:
            os.remove(self.filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                conf = float(plates['confidence']) / 100
                if conf < self.options.get('openalpr_cmdline_min_confidence'):
                    self.logger.Debug(
                        1,'OpenALPR cmd line: discarding plate: {} because detected confidence {} is less than configured min confidence: {}'
                        .format(label, conf, self.options.get('openalpr_cmdline_min_confidence')))
                    continue
                
                x1 = round(int(plates['coordinates'][0]['x']))
                y1 = round(int(plates['coordinates'][0]['y']))
                x2 = round(int(plates['coordinates'][2]['x']))
                y2 = round(int(plates['coordinates'][2]['y']))
                labels.append(label)
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return (bbox, labels, confs)
