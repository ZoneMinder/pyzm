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
        self.min_confidence = self.options.get('object_min_confidence', 0.7)
        if self.min_confidence < 1: # Rekognition wants the confidence as 0% ~ 100%, not 0.00 ~ 1.00
            self.min_confidence *= 100

        self._rekognition = boto3.client('rekognition')
        g.logger.Debug (2, f'AWS Rekognition initialised (min confidence: {self.min_confidence}%')

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
        response = self._rekognition.detect_labels(
            Image={ 'Bytes': image_jpg },
            MinConfidence=self.min_confidence
        )
        g.logger.Debug(2, f'Detection response: {response}')

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
                g.logger.Debug(3, f"{bbox=} / {label=} / {conf=}")

                bboxes.append(bbox)
                labels.append(label)
                confs.append(conf)

        return bboxes, labels, confs, ['aws']*len(labels)

    def acquire_lock(self):
        # AWS Rekognition doesn't need locking
        pass

    def release_lock(self):
        # AWS Rekognition doesn't need locking
        pass
