import cv2
import numpy as np
import pyzm.helpers.globals as g
from pyzm.ml.yolo import YoloBase, cv2_version
from pyzm.helpers.utils import Timer


class YoloOnnx(YoloBase):
    """ONNX/Ultralytics backend for YOLO (.onnx models)."""

    def __init__(self, options={}):
        super().__init__(options, default_dim=640)
        self.is_end2end = False
        self.pre_nms_layer = None

    def _load_onnx_metadata(self):
        """Load metadata from ONNX model (labels, end2end flag, pre-NMS layer)."""
        import ast as _ast
        try:
            import onnx
            model = onnx.load(self.options.get('object_weights'))
            meta = {prop.key: prop.value for prop in model.metadata_props}

            if meta.get('end2end', '').lower() == 'true':
                self.is_end2end = True
                for i, node in enumerate(model.graph.node):
                    if (node.op_type == 'Transpose'
                            and i + 1 < len(model.graph.node)
                            and model.graph.node[i + 1].op_type == 'Split'):
                        self.pre_nms_layer = 'onnx_node!' + node.name
                        break
                if self.pre_nms_layer:
                    g.logger.Debug(1, '{}: End2end ONNX detected, will read pre-NMS layer: {}'.format(
                        self.name, self.pre_nms_layer))
                else:
                    g.logger.Error('{}: End2end ONNX detected but could not find pre-NMS layer'.format(self.name))
                    self.is_end2end = False

            if 'names' in meta:
                names_dict = _ast.literal_eval(meta['names'])
                max_id = max(int(k) for k in names_dict.keys())
                classes = [''] * (max_id + 1)
                for k, v in names_dict.items():
                    classes[int(k)] = v
                return classes
        except Exception as e:
            g.logger.Debug(1, '{}: Failed to load ONNX metadata: {}'.format(self.name, e))
        return None

    def populate_class_labels(self):
        class_file_abs_path = self.options.get('object_labels')
        # Always load metadata to detect end2end flag;
        # also use embedded labels when no object_labels file is provided
        onnx_classes = self._load_onnx_metadata()
        if not class_file_abs_path:
            if onnx_classes:
                g.logger.Debug(1, '{}: Loaded {} class labels from ONNX metadata'.format(self.name, len(onnx_classes)))
                self.classes = onnx_classes
                return
            raise ValueError('{}: No object_labels provided and ONNX metadata extraction failed'.format(self.name))
        f = open(class_file_abs_path, 'r')
        self.classes = [line.strip() for line in f.readlines()]
        f.close()

    def load_model(self):
        g.logger.Debug(1, '|--------- Loading "{}" model from disk -------------|'.format(self.name))
        t = Timer()
        weights = self.options.get('object_weights')
        g.logger.Debug(1, '{}: ONNX model detected, using readNetFromONNX'.format(self.name))
        self.net = cv2.dnn.readNetFromONNX(weights)
        diff_time = t.stop_and_get_ms()

        cv2_ver = cv2_version()
        g.logger.Debug(
            1, 'perf: processor:{} {} initialization (loading {} model from disk) took: {}'
            .format(self.processor, self.name, weights, diff_time))

        self._setup_gpu(cv2_ver)
        self.populate_class_labels()

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        self.net.setInput(blob)
        if self.pre_nms_layer:
            outs = self.net.forward(self.pre_nms_layer)
        else:
            outs = self.net.forward()

        class_ids = []
        confidences = []
        boxes = []

        # Both standard and end2end (via pre-NMS layer) produce
        # ultralytics-format output: rows of [cx, cy, w, h, class_scores...]
        # Standard shape: (1, 4+C, N) — needs transpose
        # Pre-NMS shape:  (1, N, 4+C) — already transposed
        output = outs[0] if isinstance(outs, (list, tuple)) else outs
        if output.ndim == 3:
            output = output.squeeze(0)

        # Ensure predictions are (num_predictions, 4+num_classes)
        if output.shape[0] < output.shape[1]:
            predictions = output.T
        else:
            predictions = output

        g.logger.Debug(2, '{}: ONNX output shape={}, predictions={}'.format(
            self.name, output.shape, predictions.shape))

        x_factor = Width / self.model_width
        y_factor = Height / self.model_height

        for pred in predictions:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if confidence < conf_threshold:
                continue

            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x = (cx - bw / 2) * x_factor
            y = (cy - bh / 2) * y_factor
            w = bw * x_factor
            h = bh * y_factor

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

        return class_ids, confidences, boxes
