import cv2

# Pycharm hack for intellisense
# from cv2 import cv2
from pyzm.interface import GlobalConfig

g: GlobalConfig

# Class to handle HOG based detection


class Hog:
    def __init__(self, options: Optional[dict] = None):
        global g
        g = GlobalConfig()
        if options is None:
            options = {}
        self.options = options
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.winStride = self.options.get("stride")
        self.padding = self.options.get("padding")
        self.scale = float(self.options.get("scale"))
        self.meanShift = True if int(self.options.get("mean_shift")) > 0 else False
        g.logger.debug(2, "Initializing HOG")

    def get_classes(self):
        return ["person"]

    def get_options(self):
        return self.options

    def detect(self, input_image):
        r, w = self.hog.detectMultiScale(
            input_image,
            winStride=self.winStride,
            padding=self.padding,
            scale=self.scale,
            useMeanshiftGrouping=self.meanShift,
        )
        labels = []
        classes = []
        conf = []
        rects = []

        for i in r:
            labels.append("person")
            classes.append("person")
            conf.append(1.0)
            i = i.tolist()
            (x1, y1, x2, y2) = (round(i[0]), round(i[1]), round(i[0] + i[2]), round(i[1] + i[3]))
            rects.append((x1, y1, x2, y2))

        # g.logger.Debug(f'HOG:Returning: {rects}, {labels}, {conf}')
        return rects, labels, conf
