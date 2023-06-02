# VirelAI support for ZM object detection
# Author: Isaac Connor

import io
import sys
import base64

import cv2

from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g

class VirelAI(Base):
    def __init__(self, options={}):
        self.options = options
        self.min_confidence = self.options.get('object_min_confidence', 0.7)
        if self.min_confidence < 1: # Rekognition wants the confidence as 0% ~ 100%, not 0.00 ~ 1.00
            self.min_confidence *= 100

        g.logger.Debug (2, 'VirelAI initialised (min confidence: {}%'.format(self.min_confidence))

    def detect(self, image=None):
        height, width = image.shape[:2]

        g.logger.Debug(1, '|---------- VirelAI (image: {}x{}) ----------|'.format(width,height))

        is_success, _buff = cv2.imencode('.jpg', image)
        if not is_success:
            g.logger.Warning('Was unable to convert the image to JPG')
            return [], [], [], []
        image_jpg = _buff.tobytes()

        files = {'Image': {'Bytes' :  base64.b64encode(image_jpg).decode()}}

        api_url = 'https://virel.ai';
        object_url = api_url+'/api/detect/payload'
        g.logger.Debug(2, 'Invoking virelai api with url:{} and headers={} '.format(object_url, auth_header))

        start = datetime.datetime.now()
        try:
            headers = {'Content-type': 'application/json; charset=utf-8'}
            r = requests.post(url=object_url, headers=headers, json=files)
            g.logger.Debug(2, 'R: {}'.format(r.text))
            r.raise_for_status()
        except Exception as e:
            g.logger.Error('Error during remote post: {}'.format(str(e)))
            g.logger.Debug(2, traceback.format_exc())
            raise

        diff_time = (datetime.datetime.now() - start)
        g.logger.Debug(1, 'remote detection inferencing took: {}'.format(diff_time))
        response = r.json()
        
        # Parse the returned labels
        bboxes = []
        labels = []
        confs = []  # Confidences

        for item in response['Labels']:
            if 'Instances' not in item:
                continue
            for instance in item['Instances']:
                if not 'BoundingBox' in instance or not 'Confidence' in instance:
                    continue
                label = item['Name'].lower()
                conf = instance['Confidence']/100
                bbox = (
                    round(width * instance['BoundingBox']['Left']),
                    round(height * instance['BoundingBox']['Top']),
                    round(width * (instance['BoundingBox']['Left'] + instance['BoundingBox']['Width'])),
                    round(height * (instance['BoundingBox']['Top'] + instance['BoundingBox']['Height']))
                )
                g.logger.Debug(3, 'bbox={} / label={} / conf={}'.format(bbox, label, conf))

                bboxes.append(bbox)
                labels.append(label)
                confs.append(conf)

        return bboxes, labels, confs, ['VirelAI']*len(labels)

    def get_detect_image(self, image=None):
        is_success, _buff = cv2.imencode('.jpg', image)
        if not is_success:
            g.logger.Warning('Was unable to convert the image to JPG')
            return;

        image_jpg = _buff.tobytes()

        files = {'Image': {'Bytes' :  base64.b64encode(image_jpg.tobytes()).decode()}}

        api_url = 'http://virel.ai';
        try:
            headers = {'Content-type': 'application/json; charset=utf-8'}
            display_url = api_url+'/api/display/payload'
            r = requests.post(url=display_url, headers=headers, json=files)

            image_obj = np.asarray(bytearray(r.content), dtype='uint8')
            image_obj = cv2.imdecode(image_obj, cv2.IMREAD_COLOR)
            return image_obj
        except Exception as e:
              g.logger.Error('Error during image grab: {}'.format(str(e)))
              g.logger.Debug(2, traceback.format_exc())

    def acquire_lock(self):
        # Virel.AI doesn't need locking
        pass

    def release_lock(self):
        # Virel.AI doesn't need locking
        pass
