from pyzm.helpers.Base import Base
import pyzm.helpers.globals as g

import boto3

class AwsRekognition(Base):
    def __init__(self, options={}, logger=None):
        self.options = options

        self.rek = boto3.client('rekognition')
        g.logger.Debug (2, f'AWS Rekognition initialised')

    def detect(self, image=None):
        height, width = image.shape[:2]

        g.logger.Debug(1, f'|---------- AWS Rekognition (input image: {height}x{width}) ----------|')

        bbox = []
        labels = []
        conf = []
        model_names = []

        return bbox, labels, conf, model_names
