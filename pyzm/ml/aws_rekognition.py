import io
import sys
import base64

import boto3
import cv2

from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g


class AwsRekognition(Base):
    def __init__(self, options={}, logger=None):
        self.options = options

        self._rekognition = boto3.client('rekognition')
        g.logger.Debug (2, f'AWS Rekognition initialised')

    def detect(self, image=None):
        height, width = image.shape[:2]

        g.logger.Debug(1, f'|---------- AWS Rekognition (image: {width}x{height}) ----------|')

        # Convert 'image' to Base64-encoded JPG format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        is_success, _buff = cv2.imencode(".jpg", image)
        if not is_success:
            g.logger.Warning('Was unable to conver the image to JPG')
            return [], [], [], []
        image_jpg = _buff.tobytes()

        # Call AWS Rekognition
        labels = self._rekognition.detect_labels(
            Image={ 'Bytes': image_jpg },
            MinConfidence=self.min_confidence
        )
        g.logger.Debug(2, f'Detected labels: {labels}')

        # Parse the returned labels
        bbox = []
        labels = []
        conf = []

        return bbox, labels, conf, ['aws']*len(labels)
