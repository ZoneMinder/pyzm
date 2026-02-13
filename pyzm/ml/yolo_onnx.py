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
        # Letterbox state (set per-detect call)
        self._lb_scale = 1.0
        self._lb_pad_w = 0
        self._lb_pad_h = 0

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
        cv2_ver = cv2_version()
        if cv2_ver < (4, 13, 0):
            g.logger.Warning('{}: OpenCV {} may not support all ONNX operators (e.g. TopK). '
                             'OpenCV 4.13+ is recommended for ONNX YOLOv26 models.'.format(self.name, cv2.__version__))
        g.logger.Debug(1, '{}: ONNX model detected, using readNetFromONNX'.format(self.name))
        self.net = cv2.dnn.readNetFromONNX(weights)
        diff_time = t.stop_and_get_ms()

        g.logger.Debug(
            1, 'perf: processor:{} {} initialization (loading {} model from disk) took: {}'
            .format(self.processor, self.name, weights, diff_time))

        self._setup_gpu(cv2_ver)
        self.populate_class_labels()

    def _letterbox(self, image):
        """Resize image with aspect ratio preserved and pad to model dimensions.

        Stores scale and padding offsets in self._lb_scale, self._lb_pad_w,
        self._lb_pad_h for coordinate de-projection in _forward_and_parse.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.model_width, self.model_height

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size with gray (114, 114, 114) — Ultralytics default
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        self._lb_scale = scale
        self._lb_pad_w = pad_w
        self._lb_pad_h = pad_h

        return padded

    def _create_blob(self, image):
        """Override base: letterbox then normalize to [0,1] BGR->RGB blob."""
        letterboxed = self._letterbox(image)
        scale = 0.00392  # 1/255
        return cv2.dnn.blobFromImage(letterboxed,
                                     scale, (self.model_width, self.model_height), (0, 0, 0),
                                     True,
                                     crop=False)

    def _forward_and_parse(self, blob, Width, Height, conf_threshold):
        self.net.setInput(blob)
        if self.pre_nms_layer:
            outs = self.net.forward(self.pre_nms_layer)
        else:
            outs = self.net.forward()

        # Output formats:
        #   Standard (non-end2end): (1, 4+C, N) with [cx, cy, w, h, cls_scores...]
        #   End2end pre-NMS layer:  (1, N, 4+C) with [x1, y1, x2, y2, cls_scores...]
        #     (the end2end graph converts cxcywh -> xyxy before the NMS ops)
        output = outs[0] if isinstance(outs, (list, tuple)) else outs
        if output.ndim == 3:
            output = output.squeeze(0)

        # Ensure predictions are (num_predictions, 4+num_classes)
        if output.shape[0] < output.shape[1]:
            predictions = output.T
        else:
            predictions = output

        g.logger.Debug(2, '{}: ONNX output shape={}, predictions={}, end2end={}'.format(
            self.name, output.shape, predictions.shape, self.is_end2end))

        # Vectorized: extract best class and confidence per prediction
        class_scores = predictions[:, 4:]
        best_class_ids = np.argmax(class_scores, axis=1)
        best_confidences = class_scores[np.arange(len(best_class_ids)), best_class_ids]

        # Filter by confidence threshold
        mask = best_confidences >= conf_threshold
        filtered = predictions[mask]
        filtered_ids = best_class_ids[mask]
        filtered_confs = best_confidences[mask]

        if len(filtered) == 0:
            return [], [], []

        s = self._lb_scale
        pw = self._lb_pad_w
        ph = self._lb_pad_h

        if self.is_end2end:
            # End2end pre-NMS: pred[0:4] = [x1, y1, x2, y2] in letterbox space
            x1 = (filtered[:, 0] - pw) / s
            y1 = (filtered[:, 1] - ph) / s
            x2 = (filtered[:, 2] - pw) / s
            y2 = (filtered[:, 3] - ph) / s
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
        else:
            # Standard: pred[0:4] = [cx, cy, w, h] in letterbox space
            cx = filtered[:, 0]
            cy = filtered[:, 1]
            bw = filtered[:, 2]
            bh = filtered[:, 3]
            x = (cx - bw / 2 - pw) / s
            y = (cy - bh / 2 - ph) / s
            w = bw / s
            h = bh / s

        class_ids = filtered_ids.tolist()
        confidences = filtered_confs.tolist()
        boxes = np.stack([x, y, w, h], axis=1).tolist()

        return class_ids, confidences, boxes
