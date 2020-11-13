from pyzm.helpers.Base import Base
import pyzm.ml.object as  ObjectDetect
import pyzm.ml.face as FaceDetect
import re
import datetime
from pyzm.helpers.Media import MediaStream
import cv2


class DetectSequence(Base):
    def __init__(self, logger=None, options={}):
        super().__init__(logger)
        self.sequence = options.get('sequence', 'object').split(',')
        self.object_model = None
        self.face_model = None
        self.alpr_model = None
        self.ml_options = options

    def load_models(self):
        for seq in self.sequence:
            if seq == 'object':
                if not self.object_model:
                    self.object_model = ObjDetect(options=self.ml_options.get(seq, {}))
            elif seq == 'face':
                if not self.face_model:
                    self.face_model = FaceDetect(options = self.ml_options.get(seq, {}))
            else: 
                self.logger.Error('Invalid model {}'.format(seq))

    def detect_stream(self, stream, options={}):
        """Implements detection on a video stream

        Args:
            stream (string): location of media (file, url or event ID)
            api (object): instance of the API if the stream need to route via ZM
            options (dict, optional): Various options that control the detection process. Defaults to {}:
            - 'download': boolean # if True, will download video before analysis
            - 'start_frame' int # Which frame to start analysis. Default 1.
            - 'frame_skip': int # Number of frames to skip in video (example, 3 means process every 3rd frame)
            - 'max_frames' : int # Total number of frames to process before stopping
            - 'pattern': string # regexp for objects that will be matched. 'strategy' key below will be applied
                         to only objects that match this pattern
            - 'frame_set': string with exact frames to use (comma separated)
            - 'strategy': string # various conditions to stop matching as below
                'first' # stop at first match (Default)
                'most' # match the frame that has the highest number of detected objects
                'most_unique' # match the frame that has the highest number of unique detected objects
            - 'resize': int # width to resize image, default 800
            Returns:
                boxes (array): list of bounding boxes for matched frame
                labels (array): list of labels for matched frame
                confidences (array): list of confidences for matched frame
                id (int): frame id of matched frame
                img (cv2 image): image grab of matched frame
                all_matches (array of objects): list of boxes,labels,confidences of all frames matched

            Note:
            The same frames are not retrieved depending on whether you set
            ``download`` to ``True`` or ``False``. When set to ``True``, we use
            OpenCV's frame reading logic and when ``False`` we use ZoneMinder's image.php function
            which uses time based approximation. Therefore, the retrieve different frame offsets, but I assume
            they should be reasonably close.
            
        """

        

        stream_options = options
        match_strategy = options.get('strategy', 'first')
        match_pattern = re.compile(options.get('pattern', '.*'))
        all_matches = []
        matched_b = []
        matched_l = []
        matched_c = []
        matched_frame_id = None
        matched_frame_img = None
        manual_locking = False

        if len(self.sequence) > 1:
            manual_locking = False
            self.logger.Debug(1,'Using automatic locking as we are switching between models')
        else:
            manual_locking = True
            self.logger.Debug(1,'Using manual locking as we are only using one model')
            for seq in self.sequence:
                self.ml_options[seq]['auto_lock'] = False
            

        start = datetime.datetime.now()
        media = MediaStream(stream,'video', stream_options )
        while media.more():
            frame = media.read()
            if frame is None:
                self.logger.Debug(1,'Ran out of frames to read')
                break
            #fname = '/tmp/{}.jpg'.format(media.get_last_read_frame())
            #print (f'Writing to {fname}')
            #cv2.imwrite( fname ,frame)
            b = []
            l = []
            c = []
            for seq in self.sequence:
                self.logger.Debug(3,'Running sequence: {}'.format(seq))
                if seq == 'object':
                    if not self.object_model:
                        self.object_model = ObjectDetect.Object(options=self.ml_options.get(seq, {}))
                        if manual_locking:
                            self.object_model.acquire_lock()
                    _b,_l,_c, = self.object_model.detect(image=frame)
                    if (len(_l)):
                        b.append(_b)
                        l.append(_l)
                        c.append(_c)
                elif seq == 'face':
                    if not self.face_model:
                        self.face_model = FaceDetect.Face(options = self.ml_options.get(seq, {}))
                        if manual_locking:
                            self.face_model.acquire_lock()
                    _b,_l,_c, = self.face_model.detect(image=frame)
                    if (len(_l)):
                        b.extend(_b)
                        l.extend(_l)
                        c.extend(_c)
                  #  print (f'FACE: {b} {l} {c}')

           
                #print ('LABELS {} BOXES {}'.format(l,b))
                f_b = []
                f_l = []
                f_c = []

                match = list(filter(match_pattern.match, _l))
                for idx, label in enumerate(_l):
                    if label not in match:
                        continue
                    f_b.append(_b[idx])
                    f_l.append(label)
                    f_c.append(_c[idx])
                l = f_l
                b = f_b
                c = f_c

                if not len(l):
                    continue
                #print ('Frame:{}, labels:{}'.format(frames_processed, l))
                all_matches.append (
                    {
                        'frame_id': media.get_last_read_frame(),
                        'boxes': b,
                        'labels': l,
                        'confidences': c
                    }
                )
                if  ((match_strategy == 'first') 
                    or ((match_strategy == 'most') and (len(l) > len(matched_l))) 
                    or ((match_strategy == 'most_unique') and (len(set(l)) > len(set(matched_l))))):
                    matched_b = b
                    matched_c = c
                    matched_l = l
                    matched_frame_id = media.get_last_read_frame()
                    matched_frame_img = frame
                
                if (match_strategy=='first'):
                    break
            
        # release resources
        diff_time = (datetime.datetime.now() - start).microseconds / 1000

       
        if manual_locking:
            if self.face_model:
                self.face_model.release_lock()
            if self.object_model:
                self.object_model.release_lock()
        self.logger.Debug(
            1,'detection (with image loads) took: {} milliseconds to process {}'.format(diff_time, stream))
        media.stop()
        return matched_b, matched_l, matched_c, matched_frame_id, matched_frame_img, all_matches


        