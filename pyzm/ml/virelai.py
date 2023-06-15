# VirelAI support for ZM object detection
# Author: Isaac Connor

import io
import sys
import base64
import time

import requests
import cv2

from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g

class VirelAI(Base):
    def __init__(self, options={}):
        self.options = options
        self.min_confidence = float(self.options.get('object_min_confidence', 0.5))
        g.logger.Debug (2, 'VirelAI initialised (min confidence: {}'.format(self.min_confidence))

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
        auth_header = None
        g.logger.Debug(2, 'Invoking virelai api with url:{} and headers={} '.format(object_url, auth_header))

        start = time.perf_counter()
        try:
            headers = {'Content-type': 'application/json; charset=utf-8'}
            r = requests.post(url=object_url, headers=headers, json=files)
            g.logger.Debug(2, 'R: {}'.format(r.text))
            r.raise_for_status()
        except Exception as e:
            g.logger.Error('Error during remote post: {}'.format(str(e)))
            raise
        
        g.logger.Debug(1, 'remote detection inferencing took: {} s'.format(time.perf_counter() - start))
        # Parse the returned labels
        bboxes = []
        labels = []
        confs = []  # Confidences
        model_name = 'VirelAI:'
        try:
            response = r.json()
        except json.JSONDecodeError as e:
            g.logger.Error(f"Error decoding virelai api response: {e}")
        else:
            g.logger.Debug(2, f"{model_name} detection response -> {response}")
            # Parse the returned labels
            model_name = f"{model_name}:{repr(response['LabelModelVersion'])}"
            for item in response["Labels"]:
                # {
                # "LabelModelVersion": "detect-1",
                # "Img": "",
                # "Labels": [
                #   {"Confidence": "78.06", "Name": "person"},
                #   {"Confidence": "75.35", "Name": "person"}
                #   ]
                # }
                conf = float(item["Confidence"]) / 100
                if conf < float(self.min_confidence):
                    g.logger.Debug(1, f"{model_name}: confidence={conf} - min conf threshold={self.min_confidence}")
                    continue
                label = item["Name"].casefold()
                # Virel.ai does not return bounding box coords yet.
                # box = item["BoundingBox"]

                # bbox = (
                #     round(w * box["Left"]),
                #     round(h * box["Top"]),
                #     round(w * (box["Left"] + box["Width"])),
                #     round(h * (box["Top"] + box["Height"])),
                # )
                # return false bbox data for now
                bbox = (0, 0, 0, 0)
                bboxes.append(bbox)
                labels.append(label)
                confs.append(conf)
                g.logger.Debug(3, 'bbox={} / label={} / conf={}'.format(bbox, label, conf))
                
        return bboxes, labels, confs, [model_name]*len(labels)

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

    def acquire_lock(self):
        # Virel.AI doesn't need locking
        pass

    def release_lock(self):
        # Virel.AI doesn't need locking
        pass
