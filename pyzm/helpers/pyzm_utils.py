"""
pyzm_utils
======
Set of utility functions
"""
import datetime
import json
import os
import re
import time
from ast import literal_eval
from configparser import ConfigParser
from inspect import getframeinfo, stack
from pathlib import Path
from pickle import load as pickle_load, dump as pickle_dump
from random import choice
from shutil import which
from string import ascii_letters, digits
from traceback import format_exc
from typing import Optional, Union

import cv2
import numpy as np

# Pycharm hack for intellisense
# from cv2 import cv2
from pyzm.ZMLog import ZMLog
from pyzm.api import ZMApi
from pyzm.interface import GlobalConfig

# from pyfcm import FCMNotification

g: GlobalConfig = GlobalConfig()
ZM_INSTALLED: Optional[str] = which("zmdc.pl")


class FCMsend:
    """Send an FCM push notification to the zmninja App (native push notification)"""

    default_fcm_per_month: int = 8000
    default_fcm_v1_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJnZW5lcmF0b3IiOiJwbGlhYmxlIHBpeGVscyIsImlhdCI6" \
                         "MTYwMTIwOTUyNSwiY2xpZW50Ijoiem1uaW5qYSJ9.mYgsZ84i3aUkYNv8j8iDsZVVOUIJmOWmiZSYf15O0zc"
    default_fcm_v1_url = "https://us-central1-ninja-1105.cloudfunctions.net/send_push"
    tokens_file = "/var/lib/zmeventnotification/push/tokens.txt"
    # Perl localtime[4] corresponds to a month between 0-11
    int_to_month = {
        0: "January",
        1: "February",
        2: "March",
        3: "April",
        4: "May",
        5: "June",
        6: "July",
        7: "August",
        8: "September",
        9: "October",
        10: "November",
        11: "December",
        # reversed
        "January": 0,
        "February": 1,
        "March": 2,
        "April": 3,
        "May": 4,
        "June": 5,
        "July": 6,
        "August": 7,
        "September": 8,
        "October": 9,
        "November": 10,
        "December": 11,
    }

    def get_tokens(self):
        lp: str = "fcm:read tokens:"
        try:
            with open(self.tokens_file, "r") as f:
                fcm_tokens = f.read()
            fcm_tokens = json.loads(fcm_tokens)
        except Exception as fcm_load_exc:
            g.logger.error(f"{lp} failed to load tokens.txt into valid JSON: {fcm_load_exc}")
            return None
        else:
            return fcm_tokens

    def __init__(
            self,
            event_cause: str,
            tokens_file: Optional[str] = None,
            max_fcm: Optional[int] = None,
            fcm_key: Optional[str] = None,
            fcm_url: Optional[str] = None,
    ):
        """
        Initialize the FCM object

        :param event_cause: The cause of the event, will be the BODY of the notification
        :param tokens_file: Absolute path to tokens.txt
        :param max_fcm: Maximum amount of FCM invocations per month per token
        :param fcm_key: Key for FCM V1
        :param fcm_url: URL to send request to
        """

        def _check_same_month(month_: str, set_month_: str):
            return month_ != set_month_

        self.tokens_used = []
        self.event_cause = event_cause
        lp: str = "fcm:init:"
        if max_fcm:
            g.logger.debug(f"{lp} FCM max invocations set to the supplied value: {max_fcm}")
            self.default_fcm_per_month = max_fcm
        if fcm_key:
            g.logger.debug(f"{lp} FCM key set to the supplied value: {fcm_key}")
            self.default_fcm_v1_key = fcm_key
        if fcm_url:
            g.logger.debug(f"{lp} FCM url set to the supplied value: {fcm_url}")
            self.default_fcm_v1_url = fcm_url
        if tokens_file:
            g.logger.debug(f"{lp} FCM tokens file set to the supplied value: {tokens_file}")
            self.tokens_file = tokens_file

        if Path(self.tokens_file).exists() and Path(self.tokens_file).is_file():
            g.logger.debug(f"{lp} reading tokens.txt")
            fcm_tokens = self.get_tokens()
            monlist: Union[str, list] = ""
            intlist: Union[str, list] = ""
            mon_int: tuple
            zip_int_mon: zip
            send_fcm: bool = False
            total_sent: int = 0
            token_data_copy: dict = dict(fcm_tokens)
            if fcm_tokens and len(fcm_tokens["tokens"]):
                fcm_tokens = fcm_tokens["tokens"]
                for token in fcm_tokens:
                    g.logger.debug(f"DEBUG>>> STARTING TOKEN LOOP <<<DEBUG")
                    if token and token not in self.tokens_used:
                        self.tokens_used.append(token)
                        monlist = [int(mon) for mon in fcm_tokens[token]["monlist"].split(",")]
                        intlist = [int(c_down) for c_down in fcm_tokens[token]["intlist"].split(",")]
                        fcm_month = fcm_tokens[token]["invocations"]["at"]
                        fcm_month = self.int_to_month[int(fcm_month)]
                        curr_month = datetime.datetime.now().strftime("%B")
                        total_sent = int(fcm_tokens[token]["invocations"]["count"])
                        platform = fcm_tokens[token]["platform"]
                        self.app_version = fcm_tokens[token]["appversion"]
                        fcm_pkl_path = Path(f"{g.config.get('base_data_path')}/push/FCM-{token}.pkl")
                        if str2bool(fcm_tokens[token]["pushstate"]):
                            # pushstate is enabled, now check if the monitor is in the monlist
                            if g.mid in monlist:
                                # check the intlist for 'cool down'
                                zip_int_mon = zip(monlist, intlist)
                                for mon_int in zip_int_mon:
                                    g.logger.debug(
                                        f"DEBUG>>>> STARTING ITERATION THROUGH MON_INT LIST {mon_int=} <<<<DEBUG")
                                    # (mid, cooldown)
                                    if g.mid != mon_int[0]:
                                        continue
                                    g.logger.debug(
                                        f"DEBUG>>> MADE IT THROUGH THE MID CHECKER {g.mid=} - {mon_int[0]=} <<<DEBUG")
                                    if mon_int[1] == 0:
                                        # cool down is disabled, check if we are over the count for this token
                                        if _check_same_month(curr_month, fcm_month):
                                            g.logger.info(
                                                f"{lp} resetting FCM count as month has changed from {fcm_month} "
                                                f"to {curr_month}"
                                            )
                                            total_sent = 0
                                            fcm_month = curr_month
                                        else:
                                            if self._check_invocations(total_sent):
                                                # todo write the data out to tokens.txt
                                                send_fcm = True
                                                total_sent += 1
                                                continue
                                            else:
                                                g.logger.error(
                                                    f"{lp} token {token[:-10]} has exceeded the max FCM invocations per "
                                                    f"month ({self.default_fcm_per_month}, not sending FCM"
                                                )
                                    else:
                                        g.logger.debug(
                                            f"{lp} token {token[:-10]} has a cooldown of {mon_int[1]}, checking..."
                                        )
                                        # cool down is enabled, read pickled data and compare datetimes
                                        fcm_pkl: Optional[datetime] = None
                                        if fcm_pkl_path.exists():
                                            with fcm_pkl_path.open("rb") as f:
                                                fcm_pkl = pickle_load(f)
                                        if fcm_pkl:
                                            cooldown_ = (datetime.datetime.now() - fcm_pkl).total_seconds()
                                            if cooldown_ > mon_int[1]:
                                                g.logger.debug(
                                                    f'{lp} token {token[:-10]} has exceeded the cooldown wait '
                                                    f'({mon_int[1]}), sending FCM - ELAPSED: {cooldown_}')
                                                # cool down has expired,
                                                send_fcm = True
                                                continue
                                            else:
                                                g.logger.debug(
                                                    f"{lp} token {token[:-10]} has not exceeded the cooldown of "
                                                    f"{mon_int[1]}, not sending FCM - ELAPSED: {cooldown_}")
                                if send_fcm:
                                    send_fcm = False
                                    self.send_fcm(token=token, platform=platform, pkl_path=fcm_pkl_path)

                            else:
                                g.logger.debug(f"{lp} monitor {g.mid} is not in the monlist for token {token[:-10]}")
                        else:
                            g.logger.info(f"{lp} token {token[:-10]} pushstate is disabled, not sending FCM")
                # todo: write tokens.txt with updated values
                # g.logger.debug(f"{lp} updating token data for count and month if necessary")
                # old_data['tokens'][token]['invocations']['count'] = count
                # ret_month = self.int_to_month(month)
                # old_data['tokens'][token]['invocations']['at'] = ret_month
            else:
                g.logger.error(f"{lp} no tokens.txt found, not sending FCM")

    def send_fcm(self, token: str, platform: str, pkl_path: Path) -> None:
        """
        Send a FCM message to specified tokens
        :param pkl_path: Path to pickle file
        :param platform: android or ios
        :param token: The token to send the notification to
        :return: None
        """
        lp: str = 'fcm:send:'
        g.logger.info(f"{lp} sending FCM to token {token[:-30]}")
        title: str = f"{g.config.get('mon_name')} Alarm ({g.eid}) {'Ended:' if g.event_type == 'end' else ''}"
        date_fmt: str = g.config.get('fcm_date_format', ' %H:%M, %d-%b')
        body: str = f"{self.event_cause} {'ended' if g.event_type == 'end' else 'started'} at " \
                    f"{datetime.datetime.now().strftime(date_fmt)}"
        # https://portal/zm/index.php?view=image&eid=EVENTID&fid=objdetect_jpg&width=600
        image_url = f"{g.config.get('portal')}/index.php?view=image&eid={g.eid}&fid=objdetect&width=600"
        if image_url and g.config.get('user') and g.config.get('password'):
            import urllib.parse
            image_url = f"{image_url}&username={g.config.get('user')}&password=" \
                        f"{urllib.parse.quote(g.config.get('password'), safe='')}"
        fcm_log_message_id = g.config.get('fcm_log_message_id')
        fcm_log_ = str2bool(g.config.get('fcm_log_raw_message'))

        message = {
            'token': token,
            'title': title,
            'body': body,
            # 'image_url': self.image_url,
            'sound': 'default',
            # 'badge': int(self.badge),
            'log_message_id': fcm_log_message_id,
            'data': {
                'mid': g.mid,
                'eid': g.eid,
                'notification_foreground': 'true'
            }
        }
        replace_push_messages = str2bool(g.config.get('fcm_replace_push_messages'))
        android_ttl = g.config.get('fcm_android_ttl')
        android_priority = g.config.get('fcm_android_priority', 'high')
        if image_url:
            message['image_url'] = image_url
            g.logger.debug(f"DEBUG>>>> IMAGE URL = {image_url} <<<<DEBUG")
        if platform == 'android':
            message['android'] = {
                'icon': 'ic_stat_notification',
                'priority': android_priority,
            }
            if android_ttl:
                message['android']['ttl'] = android_ttl
            if replace_push_messages:
                message['android']['tag'] = 'zmninjapush'
            if self.app_version and self.app_version != 'unknown':
                g.logger.debug(f"{lp} setting channel to zmninja")
                message['android']['channel'] = 'zmninja'
            else:
                g.logger.debug(f"{lp} legacy client, NOT setting channel to zmninja")
        elif platform == 'ios':
            message['ios'] = {
                'thread_id': 'zmninja_alarm',
                'headers': {
                    'apns-priority': '10',
                    'apns-push-type': 'alert',
                    # 'apns-expiration': '0'
                }
            }
            if replace_push_messages:
                message['ios']['headers']['apns-collapse-id'] = 'zmninjapush'
        else:
            g.logger.error(f"{lp} platform {platform} is not supported!")
            return
        if fcm_log_:
            message['log_raw_message'] = 'yes'
            g.logger.debug(
                f"{lp} The server cloud function at {self.default_fcm_v1_url} will log your full message. "
                f"Please ONLY USE THIS FOR DEBUGGING and turn off later")

        # send the message with header auth
        headers = {
            'content-type': 'application/json',
            'Authorization': self.default_fcm_v1_key,
        }
        from requests import post
        response_ = post(self.default_fcm_v1_url, data=json.dumps(message), headers=headers)
        if response_ and response_.status_code == 200:
            g.logger.debug(f"{lp} FCM sent successfully to token {token[:-10]} - response message: {response_.text}")
            if pkl_path:
                g.logger.debug(f"{lp} serializing datetime object to {pkl_path}")
                try:
                    with pkl_path.open("wb") as f:
                        pickle_dump(datetime.datetime.now(), f)
                except Exception as e:
                    g.logger.error(f"{lp} failed to serialize datetime object to {pkl_path}")

        elif response_:
            g.logger.error(f"{lp} FCM failed to send to token {token[:-10]} with error {response_.status_code} - "
                           f"response message: {response_.text}")
            if response_.text.find('not a valid FCM') > -1 or response_.text.find('entity was not found') > -1:
                # todo remove the token from the file
                g.logger.warning(f"{lp} removing token {token[:-10]} from the file - NOT ACTUALLY BUT THIS WILL "
                                 f"BE IMPLEMENTED LATER")

    def _check_invocations(self, count: int):
        """Check if we have exceeded the max FCM invocations per month"""
        # "invocations": {"at":1, "count":0}
        if count < self.default_fcm_per_month:
            return True
        return False


class Timer:
    """A timer class that returns a time period in milliseconds"""

    def __init__(self, start_timer: bool = True):
        self.final_inference_time: Union[int, float] = 0
        self.started: bool = False
        self.start_time: Optional[time.perf_counter] = None
        if start_timer:
            self.start()

    def restart(self):
        self.start()

    def start(self):
        self.start_time = time.perf_counter()
        self.started = True
        self.final_inference_time = 0

    def stop(self):
        self.started = False
        self.final_inference_time = time.perf_counter() - self.start_time

    def get_ms(self) -> str:
        if self.final_inference_time:
            return f"{self.final_inference_time * 1000:.2f} ms"
        else:
            return f"{(time.perf_counter() - self.start_time) * 1000:.2f} ms"

    def stop_and_get_ms(self):
        if self.started:
            self.stop()
        return self.get_ms()


def create_animation(image: Optional[np.ndarray] = None, options: Optional[dict] = None, perf: Optional[float] = None):
    """A function to create an animation in MP4 and/or GIF.


    :param np.ndarray image: The image to use as the first few frames of the animation, usually the annotated image with labels and such.
    :param dict options:
    :param perf:
    :return:
    """
    import imageio

    def timestamp_it(img, ts_, ts_h, ts_w) -> write_text:
        """Place a timestamp on a supplied image

        :param img:
        :param ts_:
        :param ts_h:
        :param ts_w:
        :return:
        """
        ts_format = ts_.get("date format", "%Y-%m-%d %h:%m:%s")
        try:
            grab_frame = int(fid) - 1
            ts_text = (
                f"{datetime.datetime.strptime(g.Frame[grab_frame].get('TimeStamp'), ts_format)}"
                if g.Frame and g.Frame[grab_frame].get("TimeStamp")
                else datetime.datetime.now().strftime(ts_format)
            )
        except IndexError:  # frame ID converted to index isn't there? make the timestamp now()
            ts_text = datetime.datetime.now().strftime(ts_format)
        else:
            if str2bool(ts_.get("monitor id")):
                ts_text = f"{ts_text} - {mon_name} ({g.mid})"
        ts_text_color = ts_.get("text color")
        ts_bg_color = ts_.get("bg color")
        ts_bg = str2bool(ts_.get("background"))
        return write_text(
            img,
            text=ts_text,
            text_color=ts_text_color,
            x=5,
            y=18,
            h=ts_h,
            w=ts_w,
            adjust=True,
            bg=ts_bg,
            bg_color=ts_bg_color,
        )

    images: Optional[list] = None  # so we only do the frame grabbing loop 1 time
    fid: Optional[int] = int(options["fid"])
    file_name: Optional[str] = options["file name"]
    ani_types: Optional[str] = g.config.get("animation_types")
    lp: str = "animation:create:"
    if isinstance(ani_types, str):
        for ani_type in ani_types.strip().split(","):
            ani_type = ani_type.lstrip(".").strip("'").lower()
            animation_file = Path(f"{file_name}.{ani_type}")
            ani_file_exists = animation_file.exists()
            if ani_file_exists and not str2bool(g.config.get("force_animation")):
                g.logger.debug(
                    f"{lp} {file_name}.{ani_type} already exists and 'force_animation' is not "
                    f"configured, skipping..."
                )
                start = g.animation_seconds
                g.animation_seconds = (datetime.datetime.now() - start).total_seconds()
                return
            image_grab_url: str = f"{g.api.portal_url}/index.php?view=image&eid={g.eid}"
            animation_retries: int = int(g.config["animation_max_tries"])
            sleep_secs: Union[str, float] = g.config["animation_retry_sleep"]
            length, fps, last_tot_frame = 0, 0, 0
            mon_name: str = ""
            fast_gif: Optional[Union[str, bool]] = str2bool(g.config.get("fast_gif"))
            buffer_seconds: int = 5
            target_fps: int = 2
            for x in range(animation_retries):
                if (
                        (not g.api_event_response)
                        and ((g.config.get("PAST_EVENT") and x == 0) or (not g.config.get("PAST_EVENT")))
                ) or (not g.config.get("PAST_EVENT") and x > 0):
                    g.Event, g.Monitor, g.Frame = g.api.get_all_event_data()
                mon_name = g.config.get("mon_name", g.Monitor["Name"])
                if g.Frame is None or g.event_tot_frames < 1:
                    g.logger.debug(
                        f"{lp} event: {g.eid} does not have any frames written into the frame buffer, "
                        f"deferring check for {sleep_secs} seconds...",
                    )
                    animation_retries -= 1
                    time.sleep(float(sleep_secs))
                    continue
                last_tot_frame = 0
                if x > 0:
                    last_tot_frame = g.event_tot_frames
                total_time = round(float(g.Frame[-1]["Delta"]))
                fps = round(g.event_tot_frames / total_time)
                fb_length_needed = fid + (fps * buffer_seconds)
                g.logger.debug(
                    f"{g.event_tot_frames=} | {fid+fps*buffer_seconds=} | {fid=} | {fps=} | "
                    f"{buffer_seconds=} | {total_time=} | {target_fps=} "
                )
                if fid < 0 or buffer_seconds < 0 or fid < fps:
                    g.logger.error(
                        f"{lp} somethings wrong! {g.event_tot_frames=} | {fid+fps*buffer_seconds=} | "
                        f"{fid=} | {fps=} | {buffer_seconds=} | {total_time=} | {target_fps=}"
                    )
                    break
                if not g.event_tot_frames >= fb_length_needed:  # Frame buffer needs to grow
                    over_by = fid + (fps * buffer_seconds) - g.event_tot_frames
                    # we know total frames wont change so reduce fid or buffer_seconds to make it work
                    if g.config.get("PAST_EVENT"):
                        g.logger.debug(
                            f"{lp}:past event: {g.eid} does not have enough frames to create the desired length "
                            f"for {ani_type} animation. Frame buffer: {g.event_tot_frames} - Anchor frame: {fid} "
                            f"- Frame buffer length required: {fb_length_needed} - Frames over: {over_by}"
                            f" -> reducing start frame by frame buffer overage ({over_by}) and trying again"
                        )
                        fid = fid - (int(over_by) + 1)
                        continue
                    else:
                        if g.event_tot_frames == last_tot_frame:
                            g.logger.debug(
                                f"{lp}:live event: {g.eid} does not have enough frames to create the desired "
                                f"length for {ani_type} animation. Frame buffer: {g.event_tot_frames} - Anchor frame: "
                                f"{fid} - Frame buffer length required: {fb_length_needed} - Frames over: {over_by} "
                                f" -> reducing start frame by frame buffer overage ({over_by}) and trying again"
                            )
                            fid = fid - (int(over_by) + 1)
                            animation_retries -= 1
                            # no sleep as tot frames didn't change from last check
                            continue
                    g.logger.debug(
                        f"{lp}:live event: {g.eid} does not have enough frames to create the desired length for"
                        f" {ani_type} animation. Frame buffer: {g.event_tot_frames} - Anchor frame: {fid} "
                        f"- Frame buffer length required: {fb_length_needed} -> trying again"
                    )
                    animation_retries -= 1
                    time.sleep(float(sleep_secs))
                    continue
                break

            if animation_retries < 1:
                g.logger.error(
                    f"{lp} failed too many times at creating a frame buffer for the {ani_type},"
                    f" skipping animation..."
                )
                if fid < 0 or buffer_seconds < 0 or fid < fps:
                    g.logger.error(f"{lp} figure something else out for this? {fid = } {buffer_seconds = } {fps = }")
                return
            # Frame buffer for animation grabbed
            start_frame = round(max(fid - (buffer_seconds * fps), 1))
            end_frame = round(min(g.event_tot_frames, fid + (buffer_seconds * fps)))
            skip = round(fps / target_fps)
            g.logger.debug(
                f"{lp}event: {g.eid} -> Frame Buffer: {g.event_tot_frames} - Anchor Frame: {fid} - "
                f"Start Frame: {start_frame} - End Frame: {end_frame} - Skipping Every {skip} Frames -  FPS: {fps}"
            )
            vid_w = int(g.config.get("animation_width"))
            if images is None:  # So we don't grab the frames over again if creating 2+ animations
                g.logger.debug(f"{lp}:event: {g.eid} frame buffer ready to create {ani_type}, grabbing frames...")
                all_grabbed_frames = []
                images = []
                start_grabbing_frames = datetime.datetime.now()
                od_frame = None
                dim = None
                ts_font_type = cv2.FONT_HERSHEY_DUPLEX

                if image is not None:  # sent objdetect.jpg; resize and timestamp if configured
                    # Resize to configured animation_width
                    o_h, o_w = image.shape[:2]
                    image = resize_image(image, vid_w, quiet=True)
                    h, w = image.shape[:2]
                    g.logger.debug(
                        f"{lp} adding objdetect.jpg as the first few frames, original dimensions of"
                        f" -> {o_h}*{o_w} -> resized image with width: {vid_w} to {h}*{w}"
                    )
                    # Timestamp each frame in the animation
                    ts_ = g.config.get("animation_timestamp", {})
                    if ts_ and str2bool(ts_.get("enabled")):
                        image = timestamp_it(image, ts_, ts_h=h, ts_w=w)
                # grab the frame ID (frametype will always be a str of a frameID -> '212')
                elif image is None and ani_type == "mp4":
                    image_grab_url = f"{image_grab_url}&fid={fid}"
                    # grab the image and decode it
                    try:
                        response = g.api.make_request(url=image_grab_url, quiet=True)
                        img = np.asarray(bytearray(response.content), dtype="uint8")
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = resize_image(img, vid_w, quiet=True)
                    except Exception as ex:
                        g.logger.error(f"{lp} ERROR when building the first frame for {ani_type} -> {ex}")
                    else:
                        ts_h, ts_w = img.shape[:2]
                        ts_ = g.config.get("animation_timestamp", {})
                        if ts_ and str2bool(ts_.get("enabled")):
                            image = timestamp_it(img, ts_, ts_h=ts_h, ts_w=ts_w)

                frame_loop = 0
                for i in range(start_frame, end_frame + 1, skip):
                    frame_loop += 1
                    image_grab_url = f"{image_grab_url}&fid={i}"
                    try:
                        response = g.api.make_request(url=image_grab_url, quiet=True)
                        img = np.asarray(bytearray(response.content), dtype="uint8")
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # resize to 'animation_width'
                        (h, w) = img.shape[:2]
                        img = resize_image(img, vid_w, quiet=True)

                        if frame_loop == 1:
                            g.logger.debug(
                                f"{lp} resizing grabbed frames from {(h, w)} to 'animation_width' -> "
                                f"{vid_w} turns into --> {img.shape[:2]}"
                            )
                        ts_ = g.config.get("animation_timestamp", {})
                        if ts_ and str2bool(ts_.get("enabled")):
                            img = timestamp_it(img, ts_, ts_h=h, ts_w=w)
                        images.append(img)
                        all_grabbed_frames.append(i)
                    except Exception as e:
                        g.logger.error(f"{lp} error during image frame grab (includes resize and timestamp): {e}")
                end_grabbing_frames = datetime.datetime.now() - start_grabbing_frames
                g.logger.debug(
                    2,
                    f"{lp} grabbed {len(all_grabbed_frames)} frames in "
                    f"{round(end_grabbing_frames.total_seconds(), 3)} sec Frame Ids: {all_grabbed_frames}",
                )

            if ani_type == "mp4":
                od_images = []
                g.logger.debug(f"{lp} MP4 requested...")
                if image is not None:
                    for i in range(4):
                        od_images.append(image)
                od_images.extend(images)
                imageio.mimwrite(f"{file_name}.mp4", od_images, format="mp4", fps=target_fps)
                mp4_file = Path(f"{file_name}.mp4")
                size = mp4_file.stat().st_size
                g.logger.debug(
                    f"{lp} saved to {mp4_file.name}, size {size / 1024 / 1024:.2f} MB, frames: {len(images)}"
                )

            elif ani_type == "gif":
                from pygifsicle import optimize as opti

                # Let's slice the right amount from images
                # GIF uses a +- 2 second buffer
                gif_buffer_seconds = 3
                if fast_gif:
                    gif_buffer_seconds = gif_buffer_seconds * 1.5
                    target_fps = target_fps * 2

                g.logger.debug(
                    f"{lp} {'fast ' if fast_gif else 'regular speed '}GIF requested...",
                )
                gif_start_frame = int(max(fid - (gif_buffer_seconds * fps), 1))
                gif_end_frame = int(min(g.event_tot_frames, fid + (gif_buffer_seconds * fps)))
                s1 = round((gif_start_frame - start_frame) / skip)
                s2 = round((end_frame - gif_end_frame) / skip)
                if s1 >= 0 and s2 >= 0:
                    if fast_gif:
                        gif_images = images[0 + s1: -s2: 2]
                    else:
                        gif_images = images[0 + s1: -s2]
                    if image is not None:
                        num = 8 if fast_gif else 4
                        for i in range(num):
                            gif_images.insert(0, image)
                    # g.logger.debug(f"{gif_buffer_seconds=} | {target_fps=} | {gif_start_frame=} | {gif_end_frame=} "
                    #                f"| sliced from {s1=} | negative {s2=}")
                    g.logger.debug(
                        f"{lp}{'fast ' if fast_gif is not None else ''}gif: sliced {s1} to"
                        f" -{s2} from a total of {len(images)}, writing to disk..."
                    )
                    g.logger.debug(
                        f"{lp}{'fast ' if fast_gif is not None else ''}gif: optimizing GIF using gifsicle"
                        " (smaller file size at the cost of image quality)"
                    )
                    start_making_gif = datetime.datetime.now()
                    raw_gif = None
                    imageio.mimwrite(f"{file_name}.gif", gif_images, format="gif", fps=target_fps)
                    gif_file = Path(f"{file_name}.gif")
                    before_opt_size = gif_file.stat().st_size
                    opti(source=f"{file_name}.gif", colors=256)
                    size = gif_file.stat().st_size
                    diff_write = round((datetime.datetime.now() - start_making_gif).total_seconds(), 2)
                    g.logger.debug(
                        f"perf:{lp}{'fast ' if fast_gif is not None else ''}gif: {diff_write} sec to optimize "
                        f"and save {ani_type} to disk -> before: {before_opt_size / 1024 / 1024:.2f} MB --> "
                        f"after optimization: {size / 1024 / 1024:.2f} MB for {len(gif_images)} frames"
                    )
                else:
                    g.logger.debug(
                        f"{lp}{'fast ' if fast_gif is not None else ''}gif: range is weird start: s1='{s1}' "
                        f"end offset: s2='-{s2}'"
                    )
    g.animation_seconds = time.perf_counter() - perf


def resize_image(img: np.ndarray, resize_w: Union[str, int], quiet: bool = True):
    """Resize a CV2 (numpy.ndarray) image using ``resize_w``"""
    lp = "resize:img:"
    if resize_w == "no":
        g.logger.debug(f"{lp} 'resize' is set to 'no', not resizing image...") if not quiet else None
    elif img is not None:
        h, w = img.shape[:2]
        try:
            resize_w = float(resize_w)
        except Exception as all_ex:
            g.logger.error(
                f"{lp} 'resize' must be set to 'no' or a number like 800 or 320.55, any "
                f"other format will cause errors (currently set to {resize_w}), not resizing image..."
            ) if not quiet else None
        else:
            aspect_ratio: float = float(resize_w) / float(w)
            dim: tuple = (int(resize_w), int(h * aspect_ratio))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            g.logger.debug(
                2,
                f"{lp} success using resize={resize_w} - original dimensions: {w}*{h}"
                f" - resized dimensions: {dim[1]}*{dim[0]}",
            ) if not quiet else None
    else:
        g.logger.debug(f"{lp} 'resize' called but no image supplied!") if not quiet else None
    return img


def pop_coco_names(file_name: str):
    """A function to read and populate a list with the 'names' of model labels"""
    ret_val: list = []
    lp: str = "coco names:"
    if Path(file_name).exists() and Path(file_name).is_file():
        g.logger.debug(f"{lp} attempting to populate COCO names using file: '{file_name}'")
        try:
            coco = open(file_name, "r")
        except Exception as exc:
            g.logger.error(f"{lp} there was an error while trying to open the ML 'names' file to populate labels")
        else:
            for line in coco:
                line = str(line).replace("\n", "")
                ret_val.append(line)
            coco.close()
            g.logger.debug(f"{lp} successfully populated {len(ret_val)} COCO labels from '{Path(file_name).name}'")
    elif not Path(file_name).exists():
        pass
    elif not Path(file_name).is_file():
        pass
    return ret_val


def do_hass(*args):
    """Function to communicate with a Home Assistant instance for info on some 'Helper' sensors"""
    from urllib3.exceptions import InsecureRequestWarning, NewConnectionError
    from urllib3 import disable_warnings
    import requests

    # turn off insecure warnings for self-signed certificates
    disable_warnings(InsecureRequestWarning)

    lp: str = "hass add-on:"
    headers: dict = {
        "Authorization": f"Bearer {g.config.get('hass_token')}",
        "content-type": "application/json",
    }
    sensor: Optional[str] = g.config.get("hass_notify")
    cooldown: Optional[Union[str, float]] = g.config.get("hass_cooldown")
    ha_url: Optional[str] = f"{g.config.get('hass_server')}/api/states/"

    # TODO: add person.<entity> logic
    resp = None
    # First check if HA is not set up and use the local backup if configured
    if not sensor and not cooldown:
        g.logger.debug(
            4,
            f"{lp} You have HomeAssistant API support for pushover enabled but have not setup any"
            f" sensors to control the sending of pushover notifications. "
            f"Set global and/or per monitor sensors to control them. Checking for local config option "
            f"'push_cooldown'",
        )

        send_push = True
        # check if push_cooldown is set
        if g.config.get("push_cooldown"):
            g.logger.debug(
                f"{lp} no homeassistant sensors configured, "
                f"using 'push_cooldown' -> {g.config.get('push_cooldown')}"
            )
            try:
                cooldown = float(g.config["push_cooldown"])
            except TypeError as ex:
                g.logger.error(f"{lp} 'push_cooldown' malformed, sending push...")
            else:
                time_since_last_push = pkl_pushover("load", mid=g.mid)
                if time_since_last_push:
                    now: datetime = datetime.datetime.now()
                    differ = (now - time_since_last_push).total_seconds()
                    if differ < cooldown:
                        g.logger.debug(f"{lp} COOLDOWN elapsed-> {differ} / {cooldown} " f"skipping notification...")
                        send_push = False
                    else:
                        g.logger.debug(f"{lp} COOLDOWN elapsed-> {differ} / {cooldown} " f"sending notification...")
                    cooldown = None
    # connect to HASS for data on the helpers
    elif sensor:
        # Toggle Helper aka On/Off
        ha_sensor_url = f"{ha_url}{sensor}"
        try:
            resp = requests.get(
                ha_sensor_url, headers=headers, verify=False
            ).json()  # strict cert checking off, encryption still works.
        except NewConnectionError as n_ex:
            g.logger.error(f"{lp} failed to make a new connection to the HASS host '{ha_url}', sending push")
            g.logger.debug(f"{lp} EXCEPTION>>> {n_ex}")
            send_push = True
        except Exception as ex:
            g.logger.error(f"{lp} error while trying to connect to Home Assistant instance {ex}")
            g.logger.debug(f"traceback -> {format_exc()}")
            send_push = True
        else:
            if resp.get("message") == "Entity not found.":
                g.logger.error(
                    f"{lp} the configured sensor -> '{sensor}' can not be found on the Home Assistant host!"
                    f" check for spelling or formatting errors!"
                )
                send_push = True
            else:
                g.logger.debug(
                    f"{lp} the Toggle Helper sensor for monitor {g.mid} has returned -> '{resp.get('state')}'"
                )
                # The sensor returns on or off, str2bool converts that to True/False Boolean
                send_push = str2bool(resp.get("state"))
    else:
        send_push = True

    if cooldown and ((sensor and (resp is not None and str2bool(resp.get("state")))) or not sensor):
        try:
            ha_cooldown_url: str = f"{ha_url}{cooldown}"
            cooldown_response: requests.Response = requests.get(ha_cooldown_url, headers=headers)
        except Exception as ex:
            g.logger.error(f"{lp} error while trying to connect to Home Assistant instance {ex}")
            send_push = True
        else:
            resp = cooldown_response.json()
            int_val: float = float(resp.get("state", 1))
            g.logger.debug(
                f"{lp} the Number Helper (cool down) sensor for monitor {g.mid} has returned -> "
                f"'{resp.get('state')}'"
            )
            time_since_last_push = pkl_pushover("load", mid=g.mid)
            if time_since_last_push:
                differ = (datetime.datetime.now() - time_since_last_push).total_seconds()
                if differ < int_val:
                    g.logger.debug(f"{lp} SKIPPING NOTIFICATION -> elapsed: {differ} " f"- maximum: {int_val}")
                    send_push = False
                else:
                    g.logger.debug(
                        f"{lp} seconds elapsed since last successful live event "
                        f"pushover notification -> {differ} - maximum: {int_val}, allowing notification"
                    )
                    send_push = True
            else:
                send_push = True
    else:  # HASS Toggle Helper for On/Off and local 'push_cooldown' for cooldown
        if g.config.get("push_cooldown"):
            g.logger.debug(
                f"{lp} there is no homeassistant integration configured for cooldown, "
                f"using config 'push_cooldown' -> {g.config.get('push_cooldown')}"
            )
            try:
                cooldown: float = float(g.config.get("push_cooldown"))
            except Exception as ex:
                g.logger.error(f"{lp} 'push_cooldown' malformed, sending push...")
                send_push = True
            else:
                time_since_last_push = pkl_pushover("load", mid=g.mid)
                if time_since_last_push:
                    differ = (datetime.datetime.now() - time_since_last_push).total_seconds()
                    if differ < cooldown:
                        g.logger.debug(f"{lp} COOLDOWN elapsed-> {differ} / {cooldown} skipping notification...")
                        send_push = False
                    else:
                        g.logger.debug(f"{lp} COOLDOWN elapsed-> {differ} / {cooldown} sending notification...")
                        send_push = True
                else:
                    send_push = True
        else:
            send_push = True
    return send_push


def id_generator(size: int = 16, chars: str = f"{ascii_letters}{digits}") -> str:
    """Generate a pseudo-random string using ASCII characters and 0-9

    :param int size: The length of the string ot return
    :param str chars: A string to be iterated for usable characters
    """
    return "".join(choice(chars) for _ in range(size))


def digit_generator(size: int = 16, digits_: str = digits) -> str:
    """Generate a pseudo-random string using characters 0-9

    :param int size: The length of the string to return
    :param str digits_: A string to be iterated for usable characters
    """
    return "".join(choice(digits_) for _ in range(size))


def de_dup(task: Union[list, str], separator: Optional[str] = None, return_str: bool = False) -> list:
    """Removes duplicates in a string or list, if string you can also pass a separator (default: ',').

    :param bool return_str: return a space seperated string instead of a list
    :param str|list task: strings or list of strings that you want duplicates removed from
    :param str separator: seperator for task if its a str
    :returns: list of de-duplicated strings
    :rtype list:
    """
    if separator is None:
        separator = ","
    ret_list = []
    if isinstance(task, str):
        # This looks cool but isn't really informative
        [ret_list.append(x.strip()) for x in task.split(separator) if x.strip() not in ret_list]
    elif isinstance(task, list):
        [ret_list.append(x) for x in task if x not in ret_list]

    return ret_list if not return_str else " ".join([str(x) for x in ret_list])


def read_config(file: str, return_object: bool = False) -> Optional[Union[dict, ConfigParser]]:
    """Returns a ConfigParser object or a dict of the file without sections split up (doesn't decode and replace
    secrets though)
    """
    lp: str = "read config:"
    config_file: ConfigParser = ConfigParser(interpolation=None, inline_comment_prefixes="#")
    try:
        with open(file) as f:
            config_file.read_file(f)
    except Exception as exc:
        g.logger.error(f"{lp} error while opening the supplied path toa  config file -> {exc}")
        return None
    else:
        if return_object:
            return config_file  # return whole ConfigParser object if requested
        config_file.optionxform = str  # converts to lowercase strings, so MQTT_PASSWORD is now mqtt_password, etc.
        return config_file._sections  # return a dict object that removes sections and is strictly { option: value }


def write_text(
        frame: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        text_color: tuple = (0, 0, 0),
        x: Optional[int] = None,
        y: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
        adjust: bool = False,
        font: cv2 = None,
        font_scale: float = None,
        thickness: int = 1,
        bg: bool = True,
        bg_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Write supplied text onto an image"""
    lp: str = "image:write text:"
    if frame is None:
        g.logger.error(f"{lp} called without supplying an image")
    if font is None:
        font = cv2.FONT_HERSHEY_DUPLEX
    if font_scale is None:
        font_scale = 0.5
    if isinstance(bg_color, str):
        bg_color = literal_eval(bg_color)
    if isinstance(text_color, str):
        text_color = literal_eval(text_color)
    text_size = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]

    tw, th = text_size[0], text_size[1]
    if adjust:
        if not w or not h:
            # TODO make it enlarge also if too small
            g.logger.error(
                f"{lp} cannot auto adjust text as "
                f"{'W ' if not w else ''}{'and ' if not w and not h else ''}{'H ' if not h else ''}"
                f"not provided"
            )
        else:
            if x + tw > w:
                g.logger.debug(f"adjust needed, text would go out of frame width")
                x = max(0, x - (x + tw - w))

            if y + th > h:
                g.logger.debug(f"adjust needed, text would go out of frame height")
                y = max(0, y - (y + th - h))
    # print(f"{lp} FINAL: {x=} {y=} {th=} {tw=} {H=} {W=} topleft=({loc_x1}, {loc_y1=})")
    # cv2.rectangle(frame, (loc_x1, loc_y1), (loc_x1+tw+4,loc_y1+th+4), (0,0,0), cv2.FILLED)
    top_left = (x, y - th)
    bottom_right = (x + tw + 4, y + th)
    # print(
    #     f"write_text(): (Background) {top_left=} -- {bottom_right=} -- {font_scale=} -- {thickness=}"
    #     f" -- {bg_color=} H={h} -- W={w}"
    # )
    if str2bool(bg):
        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            bg_color,
            -1,
        )
    text_x = (x + 2, y + round(th / 2))
    # g.logger.debug(
    #     f"write_text(): (Text) x={text_x} -- {font_scale=} -- {thickness=} -- {text_color=} -- text_width="
    #     f"{text_size[0]} -- text_height={text_size[1]} ---  H={h} -- W={w}"
    # )
    cv2.putText(
        frame,
        text,
        # (loc_x1 + 2, loc_y2 - 2 + int(th/2)),
        text_x,
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_bbox(
        image: Optional[np.ndarray] = None,
        boxes: Optional[list] = None,
        labels: Optional[list] = None,
        confidences: Optional[list] = None,
        polygons: Optional[list] = None,
        box_color: Optional[list] = None,
        poly_color: tuple = (255, 255, 255),
        poly_thickness: int = 1,
        write_conf: bool = True,
        errors=None,
        write_model=False,
        models=None,
):
    """Draw a bounding box on a supplied image based upon coords supplied"""
    # FIXME: need to add scaling dependant on image dimensions
    # g.logger.debug("**************DRAW BBOX={} LAB={}".format(boxes,labels))
    if models is None:
        models = []
    if polygons is None:
        polygons = []
    if confidences is None:
        confidences = []
    if labels is None:
        labels = []
    if boxes is None:
        boxes = []
    slate_colors: list = [
        (39, 174, 96),
        (142, 68, 173),
        (0, 129, 254),
        (254, 60, 113),
        (243, 134, 48),
        (91, 177, 47),
    ]
    # if no color is specified, use my own slate
    # opencv is BGR
    bgr_slate_colors = slate_colors[::-1] if box_color is None else box_color

    # first draw the polygons, if configured
    w, h = image.shape[:2]
    image = image.copy()
    lp = f"image:draw bbox:"
    if poly_thickness:
        for ps in polygons:
            try:

                cv2.polylines(
                    image,
                    [np.asarray(ps["value"])],
                    True,
                    poly_color,
                    thickness=int(poly_thickness),
                )
            except Exception as exc:
                g.logger.error(f"{lp} could not draw polygon -> {exc}")
                return
    # now draw object boundaries
    arr_len = len(bgr_slate_colors)
    for i, label in enumerate(labels):
        # =g.logger.Debug (1,'drawing box for: {}'.format(label))
        box_color = bgr_slate_colors[i % arr_len]
        if write_conf and confidences:
            label += f" {round(confidences[i] * 100)}%"
        if models and write_model and models[i]:
            models[i] = models[i].lower()
            label += f"[{models[i]}]"
        # draw bounding box around object
        # g.logger.debug(f"{lp} {boxes=} -------- {polygons=}")
        # g.logger.debug(f"{lp} DRAWING COLOR={box_color} RECT={boxes[i][0]},{boxes[i][1]} {boxes[i][2]},{boxes[i][3]}")
        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), box_color, 2)

        # write text
        font_thickness = 1
        font_scale = 0.6
        # FIXME: add something better than this
        if int(w) >= 720:
            # 720p+
            font_scale = 1.0
            font_thickness = 2
        if int(w) >= 1080:
            # 1080p+
            font_scale = 1.7
            font_thickness = 2
        if int(w) >= 1880:
            # 3-4k ish? +
            font_scale = 3.2
            font_thickness = 4

        idx = min(len(stack()), 1)
        caller = getframeinfo(stack()[idx][0])
        # g.logger.debug(
        #     f"{lp} ({i + 1}/{len(labels)}) w*h={(w, h)} {font_scale=} {font_thickness=} {boxes=} "
        #     f"{poly_thickness=} {poly_color=} \n----------------------- polygon/zone area={polygons}",
        #     caller=caller
        # )
        font_type = cv2.FONT_HERSHEY_DUPLEX
        # cv2.getTextSize(text, font, font_scale, thickness)
        text_size = cv2.getTextSize(label, font_type, font_scale, font_thickness)[0]
        text_width_padded = text_size[0] + 4
        text_height_padded = text_size[1] + 4
        # print(f"DRAW BBOX - WRITE TEXT {h=} -- {w=}   {font_scale=} -- {font_thickness=} -- text_width="
        #       f"{text_size[0]} -- text_height={text_size[1]}")

        r_top_left = (boxes[i][0], boxes[i][1] - text_height_padded)
        r_bottom_right = (boxes[i][0] + text_width_padded, boxes[i][1])
        cv2.rectangle(image, r_top_left, r_bottom_right, box_color, -1)
        # cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
        # location of text is bottom left
        cv2.putText(
            image,
            label,
            (boxes[i][0] + 2, boxes[i][1] - 2),
            font_type,
            font_scale,
            [255, 255, 255],
            font_thickness,
            cv2.LINE_AA,
        )

    # now draw error (filtered detections) boxes in RED if specified
    # There is also a configurable option to include the percentage based detections that were filtered out
    # The percentage filter usually removes a lot of the bulk of detections so it is disabled by default. Turn it on to
    # see ALL of the filtered out object red boxes
    if errors:
        for _b in errors:
            cv2.rectangle(image, (_b[0], _b[1]), (_b[2], _b[3]), (0, 0, 255), 1)

    return image


def str2tuple(string):
    """Convert a string of Polygon points '123,456 789,012' to a list of int filled tuples [(123,456), (789,012)]"""
    return [tuple(map(int, x.strip().split(","))) for x in string.split(" ")]


def str_split(my_str: str, seperator: Optional[str] = None) -> list:
    """Split a string using ``seperator``, if ``seperator`` is not provided a comma (',') will be used,
    returns a list with the split string

    :param str my_str: The string to split
    :param str seperator: The seperator used to split the string
    """
    if seperator is None:
        seperator = ","
    return [x.strip() for x in my_str.split(seperator)]


def str2bool(v: Optional[Union[str, bool]]) -> Union[str, bool]:
    """Convert a string to a boolean value

    .. note::
        - The string is converted to all lower case before evaluation.
        - Strings that will return True -> ("yes", "true", "t", "y", "1", "on", "ok", "okay", "da").
        - Strings that will return False -> ("no", "false", "f", "n", "0", "off", "nyet").
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    v = str(v)
    true_ret = ("yes", "true", "t", "y", "1", "on", "ok", "okay", "da", "enabled")
    false_ret = ("no", "false", "f", "n", "0", "off", "nyet", "disabled")
    if v.lower() in true_ret:
        return True
    elif v.lower() in false_ret:
        return False
    else:
        return g.logger.error(f"str2bool: '{v}' is not able to be parsed into a boolean operator")


def verify_vals(config: dict, vals: set) -> bool:
    """Verify that the list of strings in ``vals`` is contained within the dict of ``config``.


    :param dict config: containing all config values.
    :param set vals: containing strings of the name of the keys you want to match in the config
    dictionary.
    :return: True or False
    :rtype bool:
    """
    ret: list = []
    for val in vals:
        if val in config:
            ret.append(val)
    return len(ret) == len(vals)


def import_zm_zones(reason: str, existing_polygons: list):
    """A function to import zones that are defined in the ZoneMinder web GUI instead of defining
    zones in the per-monitor section of the configuration file.


    :param reason:
    :param existing_polygons:
    :return:
    """
    match_reason: bool = False
    lp: str = "import zm zones:"
    if reason:
        match_reason = str2bool(g.config.get("only_triggered_zm_zones"))
    g.logger.debug(2, f"{lp} only trigger on ZM zones: {match_reason} reason for event: {reason}")
    g.api: ZMApi
    url = f"{g.api.portal_url}/api/zones/forMonitor/{g.mid}.json"
    j = g.api.make_request(url)
    # Now lets look at reason to see if we need to honor ZM motion zones
    for zone_ in j.get("zones", {}):
        # print(f"{lp} ********* ITEM TYPE {zone_['Zone']['Type']}")
        if str(zone_["Zone"]["Type"]).lower == "inactive":
            g.logger.debug(2, f"{lp} skipping '{zone_['Zone']['Name']}' as it is set to 'Inactive'")
            continue
        if match_reason:
            if not find_whole_word(zone_["Zone"]["Name"])(reason):
                g.logger.debug(
                    f"{lp}:triggered by ZM: not importing '{zone_['Zone']['Name']}' as it is not in event alarm cause"
                    f" -> '{reason}'"
                )
                continue
        g.logger.debug(
            2, f"{lp} '{zone_['Zone']['Name']}' @ [{zone_['Zone']['Coords']}] is being added to defined zones"
        )
        existing_polygons.append(
            {
                "name": zone_["Zone"]["Name"].replace(" ", "_").lower(),
                "value": str2tuple(zone_["Zone"]["Coords"]),
                "pattern": None,
            }
        )
    return existing_polygons


def pkl_pushover(action: str = "load", time_since_sent=None, mid=None):
    lp: str = "pushover:pickle:"
    pkl_path: str = f"{g.config.get('base_data_path')}/push" or "/var/lib/zmeventnotification/push"
    mon_file: str = f"{pkl_path}/mon-{mid}-pushover.pkl"
    if action == "load":
        g.logger.debug(2, f"{lp} trying to load '{mon_file}'")
        try:
            with open(mon_file, "rb") as fh:
                time_since_sent = pickle_load(fh)
        except FileNotFoundError:
            g.logger.debug(
                f"{lp} FileNotFound - no time of last successful push found for monitor {mid}",
            )
            return
        except EOFError:
            g.logger.debug(
                f"{lp} empty file found for monitor {mid}, going to remove '{mon_file}'",
            )
            try:
                os.remove(mon_file)
            except Exception as e:
                g.logger.error(f"{lp} could not delete: {e}")
        except Exception as e:
            g.logger.error(f"{lp} error Exception = {e}")
            g.logger.debug(f"Traceback: {format_exc()}")
        else:
            return time_since_sent

    elif action == "write":
        try:
            with open(mon_file, "wb") as fd:
                pickle_dump(time_since_sent, fd)
                g.logger.debug(4, f"{lp} time since sent:{time_since_sent} to '{mon_file}'")
        except Exception as e:
            g.logger.error(f"{lp} error writing to '{mon_file}', time since last successful push sent not recorded:{e}")
    else:
        g.logger.warning(f"{lp} the action supplied is unknown only 'load' and 'write' are supported")


def get_image(path: str, cause: str) -> str:
    """A function to return the most pertinent image based upon ``path`` and ``cause`` (cause AKA reason).


     .. note::
        GIF takes precedence over JPEG


    :param path: The absolute path of the directory containing images.
    :param cause: The 'reason' that caused the event.
    :return: A string containing the absolute path of an image or GIF.
    :rtype str:
    """
    prefix: Optional[str] = None
    if cause.startswith("["):
        prefix = cause.split("]")[0].strip("[")
    if os.path.exists(f"{path}/objdetect.gif"):
        return f"{path}/objdetect.gif"
    if os.path.exists(f"{path}/objdetect.jpg"):
        return f"{path}/objdetect.jpg"
    if prefix and prefix == "a":
        return f"{path}/alarm.jpg"
    return f"{path}/snapshot.jpg"


# credit: https://stackoverflow.com/a/5320179
def find_whole_word(w: str):
    """Still figuring this out BOI, hold over from @pliablepixels code
    The call parentheses are omitted due to the way this function is used, meaning, the user must use find_whole_word()

    :param str w: The word to search using the ``re`` module
    :return: IDK man
    """
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


def grab_frame_id(frame_id_str: str) -> str:
    """Removes the s- or a- from frame ID string

    :param frame_id_str: The string of a frame ID to split using '-'
    :return: string of the frame ID after the '-' if a '-' is in the string
    """
    ret_val = ""
    if len(frame_id_str.split("-")) > 1:
        ret_val = frame_id_str.split("-")[1]
    else:
        ret_val = frame_id_str
    return ret_val


def pkl(
        action: str,
        boxes: Optional[list] = None,
        labels: Optional[list] = None,
        confs: Optional[list] = None,
        event: Optional[str] = None,
):
    """Use the pickle module to save a python data structure to a file"""
    lp: str = "pickle:"
    if boxes is None:
        boxes = []
    if labels is None:
        labels = []
    if confs is None:
        confs = []
    if event is None:
        event = ""
    saved_bs, saved_ls, saved_cs, saved_event = None, None, None, None
    image_path = f"{g.config.get('base_data_path')}/images" or "/var/lib/zmeventnotification/images"
    mon_file = f"{image_path}/monitor-{g.mid}-data.pkl"
    if action == "load":
        g.logger.debug(2, f"{lp}  trying to load file: '{mon_file}'")
        try:
            with open(mon_file, "rb") as fh:
                saved_bs = pickle_load(fh)
                saved_ls = pickle_load(fh)
                saved_cs = pickle_load(fh)
                saved_event = pickle_load(fh)
        except FileNotFoundError:
            g.logger.debug(f"{lp}  no history data file found for monitor '{g.mid}'")
        except EOFError:
            g.logger.debug(f"{lp}  empty file found for monitor '{g.mid}'")
            g.logger.debug(f"{lp}  going to remove '{mon_file}'")
            try:
                os.remove(mon_file)
            except Exception as e:
                g.logger.error(f"{lp}  could not delete: {e}")
        except Exception as e:
            g.logger.error(f"{lp}  error: {e}")
            g.logger.error(f"{lp} traceback: {format_exc()}")
        if saved_bs:
            return saved_bs, saved_ls, saved_cs, saved_event
        else:
            return None, None, None, None
    elif action == "write":
        try:
            with open(mon_file, "wb") as f:
                pickle_dump(boxes, f)
                pickle_dump(labels, f)
                pickle_dump(confs, f)
                pickle_dump(event, f)
                g.logger.debug(
                    4,
                    f"{lp} saved_event:{event} saved boxes:{boxes} - labels:{labels} "
                    f"- confs:{confs} to file: '{mon_file}'",
                )
        except Exception as e:
            g.logger.error(f"{lp}  error writing to '{mon_file}' past detections not recorded, err msg -> {e}")


def get_www_user(possible_user: Optional[str] = None) -> tuple:
    """Returns a tuple of the web server (user, group)

    This function tries for iuser 'www-user', 'apache' and ``possible_user`` if supplied.

    :param str possible_user: If your system has a different web server user, this is where to supply it.
    :returns: ('webuser','webgroup')
    """
    import pwd
    import grp

    web_user = []
    web_group = []
    try:
        u_apache = pwd.getpwnam("apache")
    except Exception:
        pass
    else:
        web_user = "apache"
    try:
        u_www = pwd.getpwnam("www-data")
    except Exception:
        pass
    else:
        web_user = "www-data"
    try:
        g_apache = grp.getgrnam("apache")
    except Exception:
        pass
    else:
        web_group = "apache"
    try:
        g_apache = grp.getgrnam("www-data")
    except Exception:
        pass
    else:
        web_group = "www-data"
    if possible_user:
        try:
            g_apache = grp.getgrnam(possible_user)
        except Exception:
            pass
        else:
            web_group = possible_user

    return web_user, web_group


class Pushover:
    options: Optional[dict]
    file: Optional[dict]

    def __init__(self, pushover_url: Optional[str] = None):
        """Create a PushOver object to send pushover notifications via `request.post` to API

        :param str pushover_url: The URL to send pushover messages, default > https://api.pushover.net/1/messages.json
        """
        lp: str = "pushover:"
        if pushover_url is None:
            self.push_send_url = "https://api.pushover.net/1/messages.json"
        else:
            g.logger.debug(f"{lp} a URL was provided to send the pushover messages to -> {pushover_url}")
            self.push_send_url = pushover_url
        self.options = None
        self.file = None

    @staticmethod
    def check_config(config: dict) -> bool:
        """Confirms the existence of required keys"""
        req: tuple = ("user", "token")
        tot_reqs: int = 0
        for k, v in config.items():
            if k in req and v is not None:
                tot_reqs += 1
        if tot_reqs == len(req):
            return True
        else:
            return False

    def send(self, param_dict, files: Optional[dict] = None, record_last: bool = True):
        """Send PushOver notification

        files: (dict)
            Contains image (jpg or gif ONLY) ->
             {
            "attachment": (
             "name_file.gif" ,
              open("/my/great.jpg", rb),
               'image/jpeg')
            }



        param_dict: (dict)
            Contains configuration options ->
                      {
                    'token': 'APP_TOKEN',

                    'user': 'USER_KEY',

                    'title': 'my creative title',

                    'message': 'my message',

                    'url': 'clickable url in notification',

                    'url_title': 'Click here',

                    'sound': 'tugboat',

                    'priority': 0, # -2 = Lowest, -1 = Low, 0 = Normal, 1 = High, 2 = Emergency
                    'callback': None,  # EMERGENCY PRIORITY ONLY
                    'retry': 120,  # EMERGENCY PRIORITY ONLY
                    'expire': 3600,  # EMERGENCY PRIORITY ONLY

                    'device': 'a specific device',

                        }

        """
        lp: str = "pushover:"
        from requests import post

        self.options = param_dict

        self.file = files if files else self.file
        if not self.check_config(self.options):
            g.logger.error(
                f"{lp} you must specify at a minimum a push_key"
                f" and push_token in the config file to send pushover notifications!"
            )
            return
        try:
            r = post(self.push_send_url, data=self.options, files=self.file)
            r.raise_for_status()
            r = r.json()
        except Exception as ex:
            g.logger.error(f"{lp} sending notification data and converting response to JSON FAILED -> {ex}")
        else:
            g.logger.debug(f"{lp} PUSHOVER DEBUG>>> {r = }")
            if record_last:
                if r.get("status") == 1:
                    # pushover success
                    pkl_pushover("write", datetime.datetime.now(), mid=g.mid)
                else:
                    g.logger.error(
                        f"pushover: response from pushover API ->"
                        f" {r.json() if r.json() else '<NO RESPONSE to json(), PLACARD>'}"
                    )


def pretty_print(matched_data, remote_sanitized):
    # FIXME - this is when I was just starting to learn
    # Here we go, making it output pretty data
    mmloop = 0
    first_tight_line = 0
    if len(matched_data.get("labels")):  # display the results nicely
        for idx, (dkey, dval) in enumerate(remote_sanitized.items()):
            if not dval:
                # g.logger.debug(1,f"dval is None so skipping")
                continue
            # g.logger.debug(1, f"{dval = } {dval.__len__() = }")
            # print(f"{type(dval) = } ")
            if isinstance(dval, dict):
                for i_idx, (md_key, md_val) in enumerate(dval.items()):
                    if not md_val:
                        continue
                    if len(str(md_val)):
                        first_tight_line += 1
                        xb = True if first_tight_line == 1 else None
                        if i_idx == 0:
                            g.logger.debug("--- --- ---")
                        g.logger.debug(
                            f"'{dkey}'->  {md_key}-->{md_val}  ",
                            tight=True,
                            nl=xb,
                        )
            elif isinstance(dval, list):
                if dval.__len__() > 0:
                    for all_match in dval:
                        for i_idx, (md_key, md_val) in enumerate(all_match.items()):
                            if not md_val:
                                continue
                            if len(str(md_val)):
                                if i_idx == 0:
                                    g.logger.debug("--- --- ---")
                                first_tight_line += 1
                                xb = True if first_tight_line == 1 else None
                                g.logger.debug(
                                    f"{dkey}->  {md_key}-->{md_val}  ",
                                    tight=True,
                                    nl=xb,
                                )

                else:
                    mmloop += 1
                    if mmloop == 1:
                        g.logger.debug("--- --- ---")
                    first_tight_line += 1
                    xb = True if first_tight_line == 1 else None
                    g.logger.debug(f"{dkey}->  {dval}  ", tight=True, nl=xb)
            else:
                first_tight_line += 1
                xb = True if first_tight_line == 1 else None
                g.logger.debug(f"'{dkey}'->  {dval}  ", tight=True, nl=xb)


class LogBuffer:
    """A logger that will cache the log lines until it is purged."""

    @staticmethod
    def kwarg_parse(**kwargs) -> dict:
        caller, level, debug_level, message = None, "DBG", 1, None
        for k, v in kwargs.items():
            if k == "caller":
                caller = v
            elif k == "level":
                level = v
            elif k == "message":
                message = v
            elif k == "debug_level":
                debug_level = v
        return {
            "message": message,
            "caller": caller,
            "level": level,
            "debug_level": debug_level,
        }

    def __init__(self):
        self.buffer: Optional[list] = []

    def __iter__(self):
        if self.buffer:
            for _line in self.buffer:
                yield _line

    def pop(self):
        """Propagate the dictionary pop() method"""
        if self.buffer:
            return self.buffer.pop()

    # return length of buffer
    def __len__(self):
        if self.buffer:
            return len(self.buffer)

    def store(self, **kwargs):
        caller, level, debug_level, message = None, "DBG", 1, None
        kwargs = self.kwarg_parse(**kwargs)
        dt = time_format(datetime.datetime.now())
        if kwargs["caller"]:
            caller = kwargs["caller"]
        else:
            idx = min(len(stack()), 2)
            caller = getframeinfo(stack()[idx][0])
        message = kwargs["message"]
        level = kwargs["level"]
        debug_level = kwargs["debug_level"]
        disp_level = level
        if level == "DBG":
            disp_level = f"DBG{debug_level}"
        data_structure = {
            "timestamp": dt,
            "display_level": disp_level,
            "filename": Path(caller.filename).name,
            "lineno": caller.lineno,
            "message": message,
        }
        self.buffer.append(data_structure)

    def info(self, message, *args, **kwargs):
        level = "INF"
        if message is not None:
            self.store(
                level=level,
                message=message,
            )

    def debug(self, *args, **kwargs):
        level = "DBG"
        debug_level = 1
        message = None
        if len(args) == 1:
            debug_level = 1
            message = args[0]
        elif len(args) == 2:
            debug_level = args[0]
            message = args[1]
        if message is not None:
            self.store(level=level, debug_level=debug_level, message=message)

    def warning(self, message, *args, **kwargs):
        level = "WAR"
        if message is not None:
            self.store(level=level, message=message)

    def error(self, message, *args, **kwargs):
        level = "ERR"
        if message is not None:
            self.store(level=level, message=message)

    def fatal(self, message, *args, **kwargs):
        level = "FAT"
        if message is not None:
            self.store(level=level, message=message)
        self.log_close(exit=-1)

    def panic(self, message, *args, **kwargs):
        level = "PNC"
        if message is not None:
            self.store(level=level, message=message)
        self.log_close(exit=-1)

    def log_close(self, *args, **kwargs):
        if self.buffer and len(self.buffer):
            # sort it by timestamp
            self.buffer = sorted(self.buffer, key=lambda x: x["timestamp"], reverse=True)
            for _ in range(len(self.buffer)):
                line = self.buffer.pop() if len(self.buffer) > 0 else None
                if line:
                    fnfl = f"{line['filename']}:{line['lineno']}"
                    print_log_string = (
                        f"{line['timestamp']} LOG_BUFFER[{os.getpid()}] {line['display_level']} "
                        f"{fnfl} [{line['message']}]"
                    )
                    print(print_log_string)
        if exit_ := kwargs.get("exit") is not None:
            exit(exit_)
        return


def time_format(dt_form: datetime) -> str:
    """Format a timestamp to include microseconds"""
    if len(str(float(f"{dt_form.microsecond / 1e6}")).split(".")) > 1:
        micro_sec = str(float(f"{dt_form.microsecond / 1e6}")).split(".")[1]
    else:
        micro_sec = str(float(f"{dt_form.microsecond / 1e6}")).split(".")[0]
    # pad the microseconds with appended zeros
    while len(micro_sec) < 6:
        micro_sec = f"{micro_sec}0"
    return f"{dt_form.strftime('%m/%d/%y %H:%M:%S')}.{micro_sec}"


def do_mqtt(args: dict, et: str, pred, pred_out, notes_zone, matched_data, push_image, *args_):
    """A function for threaded MQTT"""
    from pyzm.helpers.mqtt import Mqtt

    lp: str = "mqtt add-on:"
    try:
        mqtt_topic = g.config.get("mqtt_topic", "zmes")
        g.logger.debug(f"{lp} MQTT is enabled, initialising...")
        mqtt_conf = {
            "mqtt_enable": g.config.get("mqtt_enable"),
            "mqtt_force": g.config.get("mqtt_force"),
            "mqtt_broker": g.config.get("mqtt_broker"),
            "mqtt_user": g.config.get("mqtt_user"),
            "mqtt_pass": g.config.get("mqtt_pass"),
            "mqtt_port": g.config.get("mqtt_port"),
            "mqtt_topic": mqtt_topic,
            "mqtt_retain": g.config.get("mqtt_retain"),
            "mqtt_qos": g.config.get("mqtt_qos"),
            "mqtt_tls_allow_self_signed": g.config.get("mqtt_tls_allow_self_signed"),
            "mqtt_tls_insecure": g.config.get("mqtt_tls_insecure"),
            "tls_ca": g.config.get("mqtt_tls_ca"),
            "tls_cert": g.config.get("mqtt_tls_cert"),
            "tls_key": g.config.get("mqtt_tls_key"),
        }
        mqtt_obj = Mqtt(config=mqtt_conf)
        mqtt_obj.connect()
    except Exception as e:
        g.logger.error(f"{lp} error during constructing an MQTT session -> {e}")
        g.logger.debug(format_exc())
    else:
        if not args.get("file"):
            mqtt_obj.create_ml_image(args.get("eventpath"), pred)
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/picture/{g.mid}",
                retain=g.config.get("mqtt_retain"),
            )
            detection_info = json.dumps(
                {
                    "eid": args.get("eventid"),
                    "mid": g.mid,
                    "name": g.config.get("mon_name"),
                    "reason": pred_out.strip(),
                    "zone": notes_zone.strip(),
                    "cause": g.config.get("api_cause"),
                    "type": et,
                    "start_time": g.Event.get("StartTime"),
                    "past_event": g.config.get("PAST_EVENT"),
                }
            )
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/data/{g.mid}",
                message=detection_info,
                retain=g.config.get("mqtt_retain"),
            )
            det_data = json.dumps(
                {
                    "labels": matched_data.get("labels"),
                    "conf": matched_data.get("confidences"),
                    "bbox": matched_data.get("boxes"),
                    "models": matched_data.get("model_names"),
                }
            )
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/rdata/{g.mid}",
                message=det_data,
                retain=g.config.get("mqtt_retain"),
            )

        else:
            # convert image to a byte array
            push_image = cv2.imencode(".jpg", push_image)[1].tobytes()
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/picture/file",
                message=push_image,
                retain=g.config.get("mqtt_retain"),
            )
            # build this with info for the FILE
            detection_info = json.dumps(
                {
                    "file_name": args.get("file"),
                    "labels": matched_data.get("labels"),
                    "conf": matched_data.get("confidences"),
                    "bbox": matched_data.get("boxes"),
                    "models": matched_data.get("model_names"),
                    "detection_type": matched_data.get("type"),
                }
            )
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/data/file",
                message=detection_info,
                retain=g.config.get("mqtt_retain"),
            )
        mqtt_obj.close()


def mlapi_import_zones(config_obj=None):
    """A function to import ZoneMinder zones into the mlapi config"""
    # FIXME
    lp = "mlapi:import zm zones:"
    zones = g.api.zones()
    c = config_obj
    if zones:
        for zone in zones:
            type_ = str(zone.type()).lower()
            mid = zone.monitorid()
            name = zone.name()
            coords = zone.coords()
            if type_ == "inactive":
                g.logger.debug(
                    f"{lp} skipping {name} as it is not a zone which we are expecting activity, " f"type: {type_}"
                )
                continue

            if mid not in c.polygons:
                c.polygons[mid] = []

            name = name.replace(" ", "_").lower()
            g.logger.debug(2, f"{lp} IMPORTING '{name}' @ [{coords}] from monitor '{mid}'")
            c.polygons[mid].append({"name": name, "value": str2tuple(coords), "pattern": None})
        # iterate polygons and apply matching detection patterns by zone name
        for poly in c.polygons[mid]:
            if poly["name"] in c.detection_patterns:
                poly["pattern"] = c.detection_patterns[poly["name"]]
                g.logger.debug(
                    2,
                    f"{lp} overriding match pattern for zone/polygon '{poly['name']}' with: "
                    f"{c.detection_patterns[poly['name']]}",
                )
    return c


def start_logs(args: dict, type_: str = "unknown", no_signal: bool = False, **kwargs):
    """A function for threaded logger creation for ZMLog

    Setup logger and API, baredebug means DEBUG level logging but do not output to console
     this is handy if you are monitoring the log files with tail -F (or the provided es.log.<detect/base> or mlapi.log)
     otherwise you get double output.
    """
    lp: str = "start logs:"
    if args.get("debug") and args.get("baredebug"):
        g.logger.warning(f"{lp} both debug flags enabled! --debug takes precedence over --baredebug")
        args.pop("baredebug")

    if args.get("debug"):
        g.config["pyzm_overrides"]["dump_console"] = True

    if args.get("debug") or args.get("baredebug"):
        g.config["pyzm_overrides"]["log_debug"] = True
        if not g.config["pyzm_overrides"].get("log_level_syslog"):
            g.config["pyzm_overrides"]["log_level_syslog"] = 5
        if not g.config["pyzm_overrides"].get("log_level_file"):
            g.config["pyzm_overrides"]["log_level_file"] = 5
        if not g.config["pyzm_overrides"].get("log_level_debug"):
            g.config["pyzm_overrides"]["log_level_debug"] = 5
        if not g.config["pyzm_overrides"].get("log_debug_file"):
            # log levels -> 1 dbg/print/blank 0 info, -1 warn, -2 err, -3 fatal, -4 panic, -5 off
            g.config["pyzm_overrides"]["log_debug_file"] = 1

    if not ZM_INSTALLED:
        # Turn DB logging off if ZM is not installed
        g.config["pyzm_overrides"]["log_level_db"] = -5

    if type_ == "mlapi":
        log_path: str = ""
        log_name = "zm_mlapi.log"
        if not ZM_INSTALLED:
            g.logger.debug(f"{lp} Zoneminder is not installed, configuring mlapi logger")
            if g.config.get("log_user"):
                log_user = g.config["log_user"]
            if g.config.get("log_group"):
                log_group = g.config["log_group"]
            elif not g.config.get("log_group") and g.config.get("log_user"):
                # use log user as log group as well
                log_group = g.config["log_user"]
            log_path = f"{g.config['base_data_path']}/logs"
            # create the log dir in base_data_path, if it exists do not throw an exception
            Path(log_path).mkdir(exist_ok=True)

        elif ZM_INSTALLED:
            g.logger.debug(f"{lp} Zoneminder is installed, configuring mlapi logger")
            # get the system's apache user (www-data, apache, etc.....)

            log_user, log_group = get_www_user()
            # fixme: what if system logs are elsewhere?
            if Path("/var/log/zm").is_dir():
                g.logger.debug(f"{lp} TESTING! mlapi is on same host as ZM, using '/var/log/zm' as logging path")
                log_path = "/var/log/zm"
            else:
                g.logger.debug(
                    f"{lp} TESTING! mlapi is on the same host as ZM but '/var/log/zm' is not created or inaccessible, "
                    f"using the configured (possibly default) log path '{g.config['base_data_path']}/logs'"
                )
                log_path = f"{g.config['base_data_path']}/logs"
                # create the log dir in base_data_path, if it exists do not throw an exception
                Path(log_path).mkdir(exist_ok=True)

        else:
            g.logger.debug(
                f"{lp} It seems there is no user to log with, there will only be console output, if anything"
                f" at all. Configure log_user and log_group in your config file -> '{args.get('config')}'"
            )
            log_user = None
            log_group = None

        log_name = g.config.get("log_name", log_name)
        # Validate log path if supplied in args
        if args.get("log_path"):
            if args.get("log_path_force"):
                g.logger.debug(f"{lp} 'force_log_path' is enabled!")
                Path(args.get("log_path")).mkdir(exist_ok=True)
            else:
                log_p = Path(args.get("log_path"))
                if log_p.is_dir():
                    log_path = args.get("log_path")
                elif log_p.exists() and not log_p.is_dir():
                    g.logger.debug(
                        f"{lp}init: the specified 'log_path' ({log_p.name}) exists BUT it is not a directory! using "
                        f"'{log_path}'."
                    )
                elif not log_p.exists():
                    print(f"{lp} the specified 'log_path' ({log_p.name}) does not exist! using '{log_path}'.")

        g.config["pyzm_overrides"]["logpath"] = log_path
        g.config["pyzm_overrides"]["webuser"] = log_user
        g.config["pyzm_overrides"]["webgroup"] = log_group

    elif type_ == "zmes":
        log_name = "zmesdetect.log"
        if args.get("monitor_id"):
            log_name = f"zmesdetect_m{args.get('monitor_id')}"
        elif args.get("file"):
            log_name = "zmesdetect_file"
        elif g.mid:
            log_name = f"zmesdetect_m{g.mid}"
        elif args.get("from_face_train"):
            log_name = "zmes_train_face"
    else:
        log_name = args.get("logname") or "zmes_external"
    if not isinstance(g.logger, ZMLog):
        g.logger = ZMLog(
            name=log_name,
            override=g.config["pyzm_overrides"],
            globs=g,
            no_signal=no_signal,
        )


def create_api(args: dict):
    """A function for threaded ZMApi creation"""
    lp = "api create:"
    g.logger.debug(f"{lp} building ZM API Session")
    # get the api going
    api_options = {
        "apiurl": g.config.get("api_portal"),
        "portalurl": g.config.get("portal"),
        "user": g.config.get("user"),
        "password": g.config.get("password"),
        "basic_auth_user": g.config.get("basic_user"),
        "basic_auth_password": g.config.get("basic_password"),
        "logger": g.logger,  # currently, just a buffer that needs to be iterated and displayed
        "disable_ssl_cert_check": str2bool(g.config.get("allow_self_signed")),
        "sanitize_portal": str2bool(g.config.get("sanitize_logs")),
    }
    try:
        g.api = ZMApi(options=api_options)
    except Exception as e:
        g.logger.error(f"{lp} {e}")
        raise e
    else:
        # get and set the monitor id, name, eventpath
        if args.get("eventid"):
            # set event id globally first before calling api event data
            g.config["eid"] = g.eid = int(args["eventid"])
            # api call for event data
            try:
                g.Event, g.Monitor, g.Frame = g.api.get_all_event_data()
            except ValueError as err:
                if str(err) == "Invalid Event":
                    g.logger.error(f"{lp} there seems to be an error while grabbing event data, EXITING!")
                    exit(1)
                else:
                    g.logger.debug(f"{lp} there is a ValueError exception >>> {err}")
            else:
                g.config["mon_name"] = g.Monitor.get("Name")
                g.config["api_cause"] = g.Event.get("Cause")
                if not args.get("reason"):
                    args["reason"] = g.Event.get("Notes")
                g.config["mid"] = g.mid = args["monitor_id"] = int(g.Monitor.get("Id"))
                if args.get("eventpath", "") == "":
                    g.config["eventpath"] = args["eventpath"] = g.Event.get("FileSystemPath")
                else:
                    g.config["eventpath"] = args["eventpath"] = args.get("eventpath")
        g.logger.debug(f"{lp} ZM API created")
