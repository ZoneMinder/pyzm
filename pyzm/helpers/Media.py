from copy import deepcopy
from os import remove
from pathlib import Path
from time import sleep
from traceback import format_exc
from typing import Optional, Tuple, Any, Union, AnyStr, List

import cv2
# Pycharm hack for intellisense
# from cv2 import cv2
import numpy as np
import requests
from imutils.video import FileVideoStream as FVStream

from pyzm.api import ZMApi
from pyzm.helpers.pyzm_utils import grab_frameid, str2bool, resize_image
from pyzm.interface import GlobalConfig

# log prefix to add onto
lp: str = 'media:'
g: GlobalConfig


class MediaStream:
    frames_processed: int

    def __init__(
            self,
            stream=None,
            type_: Optional[str] = None,
            options=None
    ):
        if type_ is None:
            type_ = 'video'
        if not options:
            raise ValueError(f"{lp} no stream options provided!")
        global g
        g = GlobalConfig()
        self.fids_global: list[str] = []
        self.fids_skipped: list[str] = []
        # <frame id>: int : cv2.im(de?)encoded image
        self.skip_all_count: int = 0
        self.stream: str = stream
        self.type: str = type_
        self.fvs: Optional[FVStream] = None
        self.next_frame_id_to_read: int = 1
        self.last_frame_id_read: int = 0
        self.options: Optional[dict] = options
        self.more_images_to_read: bool = True
        self.frames_processed: int = 0
        self.frames_skipped: int = 0
        # For 'video' type
        self.frames_read: int = 0
        self.api: ZMApi = g.api
        self.frame_set: List[AnyStr] = []
        self.frame_set_index: int = 0
        self.frame_set_len: int = 0
        self.orig_h_w: Optional[Tuple[str, str]] = None
        self.resized_h_w: Optional[Tuple[str, str]] = None
        self.default_resize: int = 800
        self.is_deletable: bool = False
        self.allowed_attempts = None
        wait: Optional[Union[str, int]] = None
        self.fids_processed: List[AnyStr] = []
        self.last_frame_id_read: Optional[Union[int, str]] = None

        if options.get("delay"):
            try:
                delay = float(options['delay'])
            except ValueError:
                g.logger.error(f"media: the configured delay is malformed! can only be a number (x or x.xx)")
            except TypeError as t_exc:
                g.logger.error(f"media: the configured delay is malformed! can only be a number (x or x.xx)")
                g.logger.debug(t_exc)
            else:
                if delay:
                    g.logger.debug(
                        f"{lp} Delay is configured, this only applies one time - waiting for {options.get('delay')} "
                        f"seconds"
                    )
                    sleep(float(options.get("delay")))

        if isinstance(self.stream, str):
            ext = Path(self.stream).suffix
            if ext.lower() in (".jpg", ".png", ".jpeg"):
                g.logger.debug(f"{lp} The supplied stream '{self.stream}' is an image file")
                self.type = "file"
                return

        self.start_frame = int(options.get("start_frame", 1))
        self.frame_skip = int(options.get("frame_skip", 1))
        self.max_frames = int(options.get("max_frames", 0))
        self.contig_frames_before_error = int(
            options.get("contig_frames_before_error", 5)
        )
        self.frames_before_error = 0
        if (
                (isinstance(self.stream, str) and self.stream.isnumeric())
                or (isinstance(self.stream, int))
        ):
            # assume it is an event id, in which case we
            # need to access it via ZM API

            if self.options.get("pre_download"):
                # If pre_download and grabbing from API, download the images locally and then run detections
                g.logger.debug(
                    f"{lp} Download event images locally before running detections for stream: {g.eid} "
                )
                api_events = self.api.events({"event_id": int(g.eid)})

                if not api_events:
                    g.logger.error(
                        f"{lp} no such event {g.eid} found with API"
                    )
                    return
                # should only be 1 event as we sent a filter for that event in the call
                for event in api_events:
                    self.stream = event.Event.download_video(
                        download_dir=self.options.get("pre_download_dir", "/tmp/")
                    )
                    self.is_deletable = True
                    self.type = "video"

            # use API
            else:
                self.next_frame_id_to_read = int(self.start_frame)
                self.stream = f"{self.api.get_portalbase()}/index.php?view=image&eid={self.stream}"
                g.logger.debug(
                    2, f"{lp} setting 'image' as type for event -> '{g.eid}'"
                )
                self.type = "image"

        if self.options.get("frame_set"):  # process frame_set
            if isinstance(self.options.get("frame_set"), str):
                self.frame_set = self.options.get("frame_set").split(",")
            elif isinstance(self.options.get("frame_set"), list):
                self.frame_set = [str(i).strip() for i in self.options.get("frame_set")]
            else:
                g.logger.debug(
                    2,
                    f"{lp} error in frame_set format (not a string or list), setting to 'snapshot,alarm,snapshot'",
                )
                self.frame_set = ["snapshot", "alarm", 'snapshot']
            self.max_frames = self.frame_set_len = len(self.frame_set)
            if self.type == 'video' and ("alarm" in self.frame_set or "snapshot" in self.frame_set):
                print(f"inside the video thing to remove snapshot and alrm")
                # remove snapshot and alarm then see if any left before error
                self.frame_set = [
                    str(i)
                    for i in self.options.get("frame_set")
                    if i != "snapshot" or i != "alarm"
                ]
                if not self.frame_set:
                    g.logger.error(
                        f"{lp} you are using 'snapshot' or 'alarm' frame ids inside of frame_set with a VIDEO FILE"
                        f", 'snapshot' or 'alarm' are a sepcial feature of grabbing frames from the ZM API. You"
                        f"must specify actual frame ID' when detecting on a VIDEO FILE"
                    )
                    raise ValueError("WRONG FRAME_SET TYPE")

            g.logger.debug(
                2,
                f"{lp} processing a maximum of {self.frame_set_len} frame"
                f"{'s' if not self.frame_set_len == 1 else ''}-> {self.frame_set}",
            )
            self.frame_set_index = 0
            self.start_frame = self.frame_set[self.frame_set_index]

        # todo check this logic still works
        if self.type == "video":
            g.logger.debug(f"{lp}VID: starting video stream {stream}")
            f = Path(self.stream).name
            self.debug_filename = f"{f}"

            self.fvs = FVStream(self.stream).start()
            sleep(0.5)  # let it settle?
            g.logger.debug(
                f"{lp}VID: first load - skipping to frame {self.start_frame}"
            )
            if self.frame_set:
                while self.fvs.more() and self.frames_read <= self.frame_set_len:
                    self.fvs.read()
                    self.frames_read += 1
            else:
                while self.fvs.more() and self.next_frame_id_to_read < int(
                        self.start_frame
                ):
                    self.fvs.read()
                    self.next_frame_id_to_read += 1

        else:  # Use API
            g.logger.debug(2, f"{lp} using API calls for stream -> {stream}")

    def get_debug_filename(self):
        return self.debug_filename

    def image_dimensions(self):
        return {"original": self.orig_h_w, "resized": self.resized_h_w}

    def get_last_read_frame(self):
        return self.last_frame_id_read

    def more(self):

        if self.type == "file":
            return self.more_images_to_read

        if self.frame_set:
            # returns true if index is less than allowable max processed frames?
            return self.frame_set_index < self.max_frames
        if self.frames_processed >= self.max_frames:
            g.logger.debug(
                f"media: bailing as we have read {self.frames_processed} frames out of an "
                f"allowed max of {self.max_frames}"
            )
            return False
        else:
            if self.type == "video":
                return self.fvs.more()
            else:
                return self.more_images_to_read

    def stop(self):

        if self.type == "video":
            self.fvs.stop()
        if self.is_deletable:
            try:
                remove(self.stream)
            except Exception as e:
                g.logger.error(f"media:stop: could not delete downloaded images: {self.stream}")
                g.logger.debug(e)
            else:
                g.logger.debug(2, f"media:stop: deleted '{self.stream}'")

    def get_last_frame(self) -> Optional[str]:
        last_frame = None
        if self.fids_global:
            last_frame = self.fids_global[-1]
        if last_frame == self.frame_set[self.frame_set_index]:
            last_frame = ""
        return str(last_frame)

    def val_type(self, cls, k, v, msg=None) -> Any:
        """return *v* converted to class type *cls* object, msg is the ValueError message to display
        if v cannot be converted to cls"""

        if v is None or not v:
            return v
        try:
            ret_val = cls(v)
        except Exception as exc:
            g.logger.error(f"media: error trying to convert '{k}' to a {repr(cls)} -> {exc}")
            if msg:
                raise ValueError(msg)
        else:
            return ret_val

    def _increment_read_frame(self, image=None, append_fid=None, l_frame_id=None):
        """Increment frame set index and frames processed. Both frame set and not frame set compatible"""
        if self.frame_set:
            self.last_frame_id_read = l_frame_id if l_frame_id else self.frame_set[self.frame_set_index]
            self.frame_set_index += 1
            if self.frame_set_index < self.frame_set_len:
                self.frames_processed += 1
                if image is None:
                    return self.read()
                else:
                    self.fids_processed.append(append_fid)
                    self.fids_global.append(append_fid)
            else:
                self.more_images_to_read = False
                self.next_frame_id_to_read = 0
                if image is None:
                    return None
                else:
                    self.fids_processed.append(append_fid)
                    self.fids_global.append(append_fid)
        elif not self.frame_set:
            self.last_frame_id_read = self.next_frame_id_to_read
            self.next_frame_id_to_read += self.frame_skip
            if image is None:
                return self.read()
            else:
                self.fids_processed.append(append_fid)
                self.fids_global.append(append_fid)
        if image.any():
            return image

    def skip_frame(self, current_frame):
        if self.frame_set:
            self.last_frame_id_read = current_frame
            self.frame_set_index += 1
            self.frames_skipped += 1
            self.fids_skipped.append(current_frame)
            self.fids_global.append(current_frame)
            if self.frame_set_index < self.frame_set_len:
                return self.read()
            else:
                return None

    def read(self):

        response: Optional[requests.Response] = None
        past_event: Optional[str] = g.config.get("PAST_EVENT")
        frame: Optional[np.ndarray] = None
        # image from file
        # 'delay_between_frames' - probably not
        if self.type == "file":
            lp: str = 'media:read:file:'
            frame = cv2.imread(self.stream)
            self.last_frame_id_read = 1
            self.orig_h_w = frame.shape[:2]
            self.frames_processed += 1
            self.more_images_to_read = False
            if self.options.get("resize", 'no') != "no":
                try:
                    self.options["resize"] = int(self.options.get("resize"))
                except TypeError:
                    g.logger.error(
                        f"{lp} 'resize' is malformed can only be a whole number (not XXX.YY only XXX) or "
                        f"'no', setting to 'no'..."
                    )
                    self.options["resize"] = "no"
                else:
                    # Get the desired width then pass it to the resize function
                    vid_w: Optional[Union[str, int]] = self.options.get("resize")
                    frame = resize_image(frame, vid_w)
            self.resized_h_w = frame.shape[:2]
            return frame

        # video from file ???
        # 'delay_between_frames' - possibly? use cases?
        elif self.type == "video":
            while True:
                lp = 'media:read:video:'
                frame = self.fvs.read()
                self.frames_read += 1

                if frame is None:
                    # this is contiguous errors (outer try) for type == 'image'
                    self.frames_before_error += 1
                    if self.frames_before_error >= self.contig_frames_before_error:
                        g.logger.error(
                            f"{lp} Error reading frame #-{self.frames_read}"
                        )
                        return
                    else:
                        g.logger.debug(
                            f"{lp} error reading frame #-{self.frames_read} -> {self.frames_before_error} of "
                            f" a max of {self.contig_frames_before_error} 'contiguous' errors"
                        )
                        continue

                self.frames_before_error = 0
                self.orig_h_w = frame.shape[:2]
                if self.frame_set and (
                        self.frames_read != int(self.frame_set[self.frame_set_index])
                ):
                    continue
                    # At this stage we are at the frame to read
                if self.frame_set:

                    self.frame_set_index += 1
                    if self.frame_set_index < len(self.frame_set):
                        g.logger.debug(4, f"{lp} now moving to frame -> {self.frame_set[self.frame_set_index]}")
                else:
                    self.last_frame_id_read = self.next_frame_id_to_read
                    self.next_frame_id_to_read += 1
                    if (self.last_frame_id_read - 1) % self.frame_skip:
                        g.logger.debug(
                            5,
                            f"{lp} skipping frame {self.last_frame_id_read}"
                        )
                        continue

                g.logger.debug(
                    2,
                    f"{lp} processing frame:{self.last_frame_id_read}",
                )
                self.fids_processed.append(
                    self.last_frame_id_read if self.last_frame_id_read != 0 else None
                )
                self.frames_processed += 1
                break
            # END OF WHILE LOOP

            g.logger.debug(
                f"{lp} ***** RESIZE={self.options.get('resize')}"
            )
            if self.options.get("resize", 'no') != "no":
                vid_w = self.options.get("resize")
                frame = resize_image(frame, vid_w)
            self.resized_h_w = frame.shape[:2]

            if str2bool(self.options.get("save_frames")) and self.debug_filename:
                d = self.options.get("save_frames_dir", "/tmp")
                if Path(d).exists() and Path(d).is_dir():
                    f_name = f"{d}/{self.debug_filename}-output_video-{self.last_frame_id_read}.jpg"
                    g.logger.debug(
                        2,
                        f"{lp}video: stream_sequence is configured to save every frame! saving image to '{f_name}'"
                    )
                    cv2.imwrite(f_name, frame)

            return frame

        # grab images frame by frame from ZM API (can pre download frames and then process)
        elif self.type == "image":
            lp = 'media:read:image:'
            delayed: bool = False
            comp_fid: Optional[str] = None
            skip_all: bool = False
            f_difference: int = 0
            delay_frames = self.val_type(
                float,
                "delay_between_frames",
                self.options.get("delay_between_frames"),
                f"{lp} error converting 'delay_between_frames' to float"
            )
            if self.frame_set_index >= self.frame_set_len:
                # todo: add the ability to add more frames to the buffer if no detections have been found
                #  with the configured frame_set.
                return None
            if not past_event and (self.frames_processed > 0 or self.frames_skipped > 0) and delay_frames:
                # does not delay the first frame read or any frames if it is a past event.
                # All frames are already written to disk
                g.logger.debug(
                    4,
                    f"{lp} 'delay_between_frames' sleeping {delay_frames} seconds "
                    f"before reading next frame"
                )
                delayed = True
                sleep(delay_frames)
            elif past_event and delay_frames:
                g.logger.debug(f"{lp} 'delay_between_frames' is configured but this is a past event, ignoring delay...")

            current_frame: str = str(self.frame_set[self.frame_set_index]).strip()
            # delay_between_snapshot, this frame and the last frame must be snapshot for the delay to activate
            if (
                    (current_frame.startswith("sn")
                     and self.get_last_frame().startswith("s")
                     and self.options.get("delay_between_snapshots"))
                    and not delayed
            ):
                g.logger.debug(
                    2,
                    f"{lp} sleeping {self.options.get('delay_between_snapshots')} seconds before "
                    f"reading concurrent snapshot frame ('delay_between_snapshots'), last frame was a snapshot and so "
                    f"is this frame request"
                )
                snapshot_sleep = self.val_type(
                    float,
                    "delay_between_snapshots",
                    self.options.get("delay_between_snapshots"),
                    f"{lp} error converting 'delay_between_snapshots' to a float"
                )
                delayed = True
                sleep(snapshot_sleep)

            if self.frames_skipped > 0 or self.frames_processed > 0:
                g.logger.debug(
                    f"{lp} [{self.frames_processed} frames processed: {self.fids_processed}] - ["
                    f"{self.frames_skipped} Frames skipped: {self.fids_skipped}] - [Requested FID: {current_frame}] "
                    f"[Last Requested Frame ID: {self.get_last_frame()}] [Maximum # of frames to process: "
                    f"{self.max_frames}]"
                )
            else:
                g.logger.debug(f"{lp} about to process first frame!")

            fb_length_before_api_call: int = len(g.Frame) if g.Frame else 0
            if (
                    not g.api_event_response
                    or not past_event
                    or not g.Frame
                    or not g.Event
            ):
                try:
                    g.Event, g.Monitor, g.Frame = self.api.get_all_event_data(g.eid)
                except Exception as e:
                    g.logger.error(f"{lp} error grabbing event data from API -> {e}")
                    raise e
                else:
                    g.logger.debug(f"{lp} grabbed event data from ZM API")

            if not g.Frame:
                g.logger.error(
                    f"{lp} There seems to be an error with the 'Frames' DataBase in ZM! Check"
                    f"that monitor {g.mid}-> {g.config.get('mon_name')} is 'Capturing' in the ZM WEB GUI"
                )
                raise ValueError(f"There is no 'Frame' data passed from ZM API (ZM DB Errors? Monitor not 'Capturing'?")

            ####################################################
            #       CONVERT SNAPSHOT / ALARM TO FRAME ID
            ####################################################
            if current_frame.startswith("a"):
                current_frame = self.frame_set[self.frame_set_index] = f"a-{g.Event.get('AlarmFrameId')}"
                g.logger.debug(
                    2,
                    f"{lp} Event: {g.eid} - converting 'alarm' to a frame ID -> {current_frame}",
                )
            elif current_frame.startswith("sn"):
                current_frame = f"s-{g.Event.get('MaxScoreFrameId')}"
                g.logger.debug(
                    2,
                    f"{lp} Event: {g.eid} - converting 'snapshot' to a frame ID -> {current_frame}",
                )
            if not past_event:
                g.logger.debug(
                    f"{lp} [Requested Frame ID: {current_frame}] - [Frame "
                    f"Buffer Length: {g.event_tot_frames}] - [Previous Frame Buffer Length: "
                    f"{fb_length_before_api_call}"
                )
            if (g.event_tot_frames and fb_length_before_api_call == g.event_tot_frames) and not past_event:
                self.skip_all_count += 1
                # the event frame buffer has not grown since last api call
                if self.skip_all_count > 1:
                    g.logger.debug(
                        f"{lp} the frame buffer ({g.event_tot_frames}) for event '{g.eid}' is no longer"
                        f" growing. All out of bound frame ID's will be skipped"
                        f" instead of being reduced to the last available frame buffer ID"
                    )
                    skip_all = True

            total_time = self.val_type(
                float,
                'total event time',
                g.Frame[-1]["Delta"],
                f"{lp} error converting 'total event time' to a float"
            )
            f_fps = self.val_type(
                float,
                'frame rate',
                g.event_tot_frames,
                f"{lp} error converting 'frame rate' to a float"
            )
            f_fps = f_fps / total_time
            # Check if we have already processed this frame ID before
            comp_fid = grab_frameid(current_frame)
            for x in self.fids_processed:
                x = grab_frameid(str(x))  # convert a-xx or s-xx to xx if needed
                if x == comp_fid:
                    g.logger.debug(
                        f"{lp} skipping frame ID: '{current_frame}' as it has already been"
                        f" processed for event {g.eid} -> processed Frame IDs: {self.fids_processed}"
                    )
                    return self.skip_frame(current_frame)

            if self.frames_processed > 0 or self.frames_skipped > 0:
                # out of bounds logic
                # todo rework to make frame_set realize if there are a good amount of frames left on disk and so far
                #  there is no detection, grab more frames at a calculated increment. Will need a way to ask
                #   detect_stream about its current detections
                oob_frame_id = self.val_type(
                    int,
                    'out of bounds frame ID',
                    grab_frameid(current_frame),
                    f"{lp} 'out of bounds frame ID' there seems to be a formatting error "
                    f"while converting {grab_frameid(current_frame)} to an int "
                )
                if (oob_frame_id >= g.event_tot_frames) and not skip_all:
                    # if frame buffer is growing and the difference between the out of bound fid and current end
                    # fid is within a reasonable distance, let the attempts happen.
                    f_difference = oob_frame_id - g.event_tot_frames
                    f_difference = self.val_type(
                        float,
                        'frames over by',
                        f_difference,
                        f"{lp} error while converting the frames over by into a float"
                    )
                    fps_thresh = self.options.get("smart_fps_thresh", 5)
                    fps_thresh = self.val_type(
                        float,
                        'smart_fps_thresh',
                        fps_thresh,
                        f"{lp} error while converting 'smart_fps_thresh' to a float"
                    )
                    thresh_frames = f_fps * fps_thresh
                    thresh_sec = round(thresh_frames / f_fps, 2)

                    g.logger.debug(
                        f"{lp} monitor {g.mid}-> '{g.config.get('mon_name')}' is running at {f_fps} FPS - "
                        f"the configured 'smart_fps_thresh' of {fps_thresh} converts to {thresh_frames} frames. "
                        f"Overage of {f_difference} frames = {thresh_sec} seconds"
                    )
                    if f_difference <= (f_fps * fps_thresh) and not past_event:
                        self.allowed_attempts = 3  # make configurable?
                        smart_val = fps_thresh / self.allowed_attempts
                        smart_val = smart_val + (0.11 * self.allowed_attempts)
                        smart_sleep = self.val_type(float, 'smart sleep', smart_val)

                        g.logger.debug(
                            f"{lp} Based on a 'smart_fps_thresh' of {fps_thresh} seconds and {f_fps} fps. "
                            f"The requested frame ID {current_frame} is within the 'smart_fps_thresh'. Setting delay "
                            f"between attempt loops at {smart_sleep} seconds and sleeping..."
                        )
                        sleep(smart_sleep)

                    else:
                        g.logger.debug(
                            f"{lp} Based on a 'smart_fps_thresh' of {fps_thresh} seconds and {f_fps} fps. "
                            f"The requested frame ID {current_frame} is outside of the 'smart_fps_thresh'. Decreasing "
                            f"frame ID to the last available frame in the available frame buffer (-1 frame for "
                            f"disk write) -> {g.event_tot_frames - 2}"
                        )
                        # convert to an index by subtracting 1 from the frame id and 1 more to allow for disk writes
                        self.last_frame_id_read = comp_fid = self.frame_set[
                            self.frame_set_index
                        ] = str(g.event_tot_frames - 2)
                elif (
                        oob_frame_id >= g.event_tot_frames
                        and skip_all
                ):
                    g.logger.debug(
                        f"{lp} skipping out of bound frame -> "
                        f"'{current_frame}' as event '{g.eid}' has not written "
                        f"anymore frames to disk since the last time the frame buffer was checked"
                    )
                    return self._increment_read_frame(l_frame_id=oob_frame_id)

            fid_url = f"{self.stream}&fid={comp_fid}"

            # outer try is contiguous errors, need a response with code 200 and to build the image ELSE error
            try:
                fid_grab_attempts = 0
                # make sure correct types
                max_attempts = self.val_type(
                    int,
                    "max_attempts",
                    self.options.get("max_attempts", 3),
                    f"{lp} error while converting 'max_attempts' to an int"
                )
                sleep_time = self.val_type(
                    float,
                    "delay_between_attempts",
                    self.options.get("delay_between_attempts", 3.0),
                    f"{lp} error while converting 'delay_between_attempts' to a float"
                )

                for x in range(max_attempts):
                    fid_grab_attempts += 1
                    if fid_grab_attempts > 1 and int(comp_fid) >= g.event_tot_frames:
                        # drop the fid down to last available or skip it if all skip
                        if skip_all:
                            g.logger.debug(
                                f"{lp}grabbing frame: the requested frame ID: {comp_fid} is out of bounds of the "
                                f"current frame buffer by {int(comp_fid) - g.event_tot_frames} frames. Skipping frame..."
                            )
                            return self.skip_frame(current_frame)
                        else:
                            g.logger.debug(
                                f"{lp}grabbing frame: the requested frame ID: {comp_fid} is out of bounds of the "
                                f"current frame buffer by {int(comp_fid) - g.event_tot_frames} frames. Setting frame "
                                f"ID to last available frame ({g.event_tot_frames - 2}) and retrying..."
                            )
                            fid_url = f"{self.stream}&fid={g.event_tot_frames - 2}"
                    # inner try is max_attempts, grabbing the frame ID from the API
                    try:
                        response = self.api.make_request(fid_url)
                    except Exception as e:
                        err_msg = f"{e}"
                        print(f"DEBUG:MEDIA!  exception={err_msg}")
                        if err_msg != "BAD_IMAGE":
                            print("BREAKING OUT OF LOOP")
                            break
                        if sleep_time and not past_event:
                            print(f"BEFORE REQUEST: it is not a past event and {sleep_time=} -- requesting event "
                                  f"data from ZM API {g.event_tot_frames=}")
                            # check if the frame buffer has grown, maybe we keep requesting a frame that
                            # will never be there. If so, drop frame ID to last available frame
                            g.Event, g.Monitor, g.Frame = self.api.get_all_event_data(event_id=g.eid)
                            print(f"AFTER REQUEST: ZM Frame Buffer has {g.event_tot_frames} frames, we want this frame "
                                  f"id -> {comp_fid}")
                            sleep(sleep_time)

                        if fid_grab_attempts >= max_attempts:
                            # only for logging
                            g.logger.debug(
                                2,
                                f"media:read:image: failed attempt {fid_grab_attempts}/{max_attempts} of grabbing the"
                                f" image for frame ID '{comp_fid}' -> skipping frame... ",
                            )
                    else:  # request worked, no need to retry
                        break
            except Exception as e:
                g.logger.debug(f"exception come and fix type so not so broad -> {e}")
                return self._increment_read_frame(l_frame_id=comp_fid)

            # api returned something?
            else:
                if response and response.status_code == 200:
                    self.frames_before_error = 0
                    try:
                        r = response
                        img = response.content
                        img = np.asarray(bytearray(img), np.uint8)
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        self.orig_h_w = img.shape[:2]
                        if str2bool(self.options.get("save_frames")):
                            d = self.options.get("save_frames_dir", '/tmp')
                            if Path(d).is_dir():
                                f_name = f"{d}/saved-{g.eid}_{comp_fid}.jpg"
                                g.logger.debug(
                                    2,
                                    f"{lp} 'save_frames' is configured! -> saving image to '{f_name}'"
                                )
                                try:
                                    cv2.imwrite(f_name, img)
                                except Exception as exc:
                                    g.logger.error(f"{lp} ERROR writing image to disk -> {exc}")
                            else:
                                g.logger.error(f"{lp} there is aproblem with the configured 'save_frames_dir' "
                                               f"check the path for spelling mistakes")

                        if self.options.get("resize", "no") != "no":
                            print(f"{self.options.get('resize') = }")
                            # fixme: why is there a default resize?
                            vid_w = int(self.options.get("resize", self.default_resize))
                            img = resize_image(img, vid_w)
                            self.resized_h_w = img.shape[:2]
                        else:
                            g.logger.debug(f"{lp} Image returned from ZM API dimensions: {self.orig_h_w}")
                    except Exception as e:
                        g.logger.error(
                            f"{lp} could not build IMAGE from response (cv2.imdecode, cv2.resize) "
                            f"URL={fid_url} ERR_MSG={e}"
                        )
                        print(format_exc())
                        return None
                    # received a reply with an image and constructed the image
                    else:
                        a_fid = str(comp_fid) if self.frame_set else self.last_frame_id_read
                        return self._increment_read_frame(image=img, append_fid=a_fid, l_frame_id=current_frame)

                # response code not 200 or no response
                else:
                    g.logger.error(
                        f"{lp} {'response status code=' if response else ''}"
                        f"{response.status_code if response else 'no response from API call'}"
                    )
                    self.frames_before_error += 1
                    if past_event:
                        g.logger.error(
                            f"{lp} 'contiguous' error grabbing frame '{comp_fid}' PAST event logic overrides "
                            f"contiguous errors -> max=1, skipping frame..."
                        )
                    elif self.frames_before_error >= self.contig_frames_before_error:
                        g.logger.error(
                            f"{lp} exhausted 'contiguous' attempts -> {self.contig_frames_before_error} to "
                            f"grab the image for frame ID '{comp_fid}' -> skipping frame..."
                        )
                    else:
                        g.logger.error(
                            f"{lp} 'contiguous' error grabbing frame '{comp_fid}' ({self.frames_before_error}/"
                            f"{self.contig_frames_before_error}), retrying frame..."
                        )
                        return self.read()

                    self._increment_read_frame(l_frame_id=comp_fid)
