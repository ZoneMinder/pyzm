"""
pyzm_utils
======
Set of utility functions
"""
import datetime
import json
import os
import re
import sys
import time
from ast import literal_eval
from configparser import ConfigParser
from inspect import getframeinfo, stack
from pickle import load as pickle_load, dump as pickle_dump
from random import choice
from string import ascii_letters, digits
from traceback import format_exc
from typing import Optional, Union
from pathlib import Path

import numpy as np
import cv2
# Pycharm hack for intellisense
# from cv2 import cv2
from pyzm.interface import GlobalConfig

g: Optional[GlobalConfig] = None


def set_g(globs):
    global g
    g = globs


class Timer:
    def __init__(self, start_timer=True):
        self.final_inference_time = 0
        self.started = False
        self.start_time = None
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

    def get_ms(self):
        if self.final_inference_time:
            return f"{self.final_inference_time * 1000:.2f} ms"
        else:
            return f"{(time.perf_counter() - self.start_time) * 1000:.2f} ms"

    def stop_and_get_ms(self):
        if self.started:
            self.stop()
        return self.get_ms()


def createAnimation(
        image=None,
        options: dict = None,
        perf=None
):
    import imageio

    def timestamp_it(img, ts_, ts_h, ts_w):
        ts_format = ts_.get('date format', '%Y-%m-%d %h:%m:%s')
        try:
            grab_frame = int(fid) - 1
            ts_text = (
                f"{datetime.datetime.strptime(g.Frame[grab_frame].get('TimeStamp'), ts_format)}"
                if g.Frame and g.Frame[grab_frame].get('TimeStamp')
                else datetime.datetime.now().strftime(ts_format)
            )
        except IndexError:  # frame ID converted to index isn't there? make the timestamp now()
            ts_text = datetime.datetime.now().strftime(ts_format)
        else:
            if str2bool(ts_.get('monitor id')):
                ts_text = f"{ts_text} - {mon_name} ({g.mid})"
        ts_text_color = ts_.get('text color')
        ts_bg_color = ts_.get('bg color')
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

    images = None  # so we only do the frame grabbing loop 1 time
    g = options['conf globals']
    fid = int(options['fid'])
    file_name = options['file name']
    ani_types = g.config.get("animation_types")
    log_prefix = "animation:create:"
    if isinstance(ani_types, str):
        for ani_type in ani_types.strip().split(","):
            ani_type = ani_type.lstrip(".").strip("'").lower()
            animation_file = Path(f"{file_name}.{ani_type}")
            does_exist = animation_file.exists()
            if does_exist and not (
                    str2bool(g.config.get("force_animation"))
            ):
                g.logger.debug(
                    f"{log_prefix} {file_name}.{ani_type} already exists and 'force_animation' isn't "
                    f"configured, skipping..."
                )
                start = g.animation_seconds
                g.animation_seconds = (datetime.datetime.now() - start).total_seconds()
                return

            image_grab_url = f"{g.config.get('portal')}/index.php?view=image&eid={g.eid}"
            animation_retries = int(g.config["animation_max_tries"])
            sleep_secs = g.config["animation_retry_sleep"]
            length, fps, last_tot_frame = 0, 0, 0
            mon_name: str = ''
            fast_gif = str2bool(g.config.get("fast_gif"))
            buffer_seconds = 5
            target_fps = 2
            for x in range(animation_retries):
                if (
                        ((not g.api_event_response)
                         and ((g.config.get("PAST_EVENT") and x == 0)
                              or (not g.config.get("PAST_EVENT"))))
                        or (not g.config.get("PAST_EVENT") and x > 0)
                ):
                    g.Event, g.Monitor, g.Frame = g.api.get_all_event_data()
                mon_name = g.config.get('mon_name', g.Monitor["Name"])
                if g.Frame is None or g.event_tot_frames < 1:
                    g.logger.debug(
                        f"{log_prefix} event: {g.eid} does not have any frames written into the frame buffer, "
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
                        f"{log_prefix} somethings wrong! {g.event_tot_frames=} | {fid+fps*buffer_seconds=} | "
                        f"{fid=} | {fps=} | {buffer_seconds=} | {total_time=} | {target_fps=}"
                    )
                    break
                if not g.event_tot_frames >= fb_length_needed:  # Frame buffer needs to grow
                    over_by = fid + (fps * buffer_seconds) - g.event_tot_frames
                    # we know total frames wont change so reduce fid or buffer_seconds to make it work
                    if g.config.get("PAST_EVENT"):
                        g.logger.debug(
                            f"{log_prefix}:past event: {g.eid} does not have enough frames to create the desired length "
                            f"for {ani_type} animation. Frame buffer: {g.event_tot_frames} - Anchor frame: {fid} "
                            f"- Frame buffer length required: {fb_length_needed} - Frames over: {over_by}"
                            f" -> reducing start frame by frame buffer overage ({over_by}) and trying again"
                        )
                        fid = fid - (int(over_by) + 1)
                        continue
                    else:
                        if g.event_tot_frames == last_tot_frame:
                            g.logger.debug(
                                f"{log_prefix}:live event: {g.eid} does not have enough frames to create the desired "
                                f"length for {ani_type} animation. Frame buffer: {g.event_tot_frames} - Anchor frame: "
                                f"{fid} - Frame buffer length required: {fb_length_needed} - Frames over: {over_by} "
                                f" -> reducing start frame by frame buffer overage ({over_by}) and trying again"
                            )
                            fid = fid - (int(over_by) + 1)
                            animation_retries -= 1
                            # no sleep as tot frames didn't change from last check
                            continue
                    g.logger.debug(
                        f"{log_prefix}:live event: {g.eid} does not have enough frames to create the desired length for"
                        f" {ani_type} animation. Frame buffer: {g.event_tot_frames} - Anchor frame: {fid} "
                        f"- Frame buffer length required: {fb_length_needed} -> trying again"
                    )
                    animation_retries -= 1
                    time.sleep(float(sleep_secs))
                    continue
                break

            if animation_retries < 1:
                g.logger.error(
                    f"{log_prefix} failed too many times at creating a frame buffer for the {ani_type},"
                    f" skipping animation..."
                )
                if fid < 0 or buffer_seconds < 0 or fid < fps:
                    g.logger.error(
                        f"{log_prefix} figure something else out for this? {fid = } {buffer_seconds = } {fps = }"
                    )
                return
            # Frame buffer for animation grabbed
            start_frame = round(max(fid - (buffer_seconds * fps), 1))
            end_frame = round(min(g.event_tot_frames, fid + (buffer_seconds * fps)))
            skip = round(fps / target_fps)
            g.logger.debug(
                f"{log_prefix}event: {g.eid} -> Frame Buffer: {g.event_tot_frames} - Anchor Frame: {fid} - "
                f"Start Frame: {start_frame} - End Frame: {end_frame} - Skipping Every {skip} Frames -  FPS: {fps}"
            )
            vid_w = int(g.config.get("animation_width"))
            if images is None:  # So we don't grab the frames over again if creating 2+ animations
                g.logger.debug(
                    f"{log_prefix}:event: {g.eid} frame buffer ready to create {ani_type}, grabbing frames..."
                )
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
                        f"{log_prefix} adding objdetect.jpg as the first few frames, original dimensions of"
                        f" -> {o_h}*{o_w} -> resized image with width: {vid_w} to {h}*{w}"
                    )
                    # Timestamp each frame in the animation
                    ts_ = g.config.get('animation_timestamp', {})
                    if (
                            ts_
                            and str2bool(ts_.get('enabled'))
                    ):
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
                        g.logger.error(f"{log_prefix} ERROR when building the first frame for {ani_type} -> {ex}")
                    else:
                        ts_h, ts_w = img.shape[:2]
                        ts_ = g.config.get('animation_timestamp', {})
                        if ts_ and str2bool(ts_.get('enabled')):
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
                                f"{log_prefix} resizing grabbed frames from {(h, w)} to 'animation_width' -> "
                                f"{vid_w} turns into --> {img.shape[:2]}"
                            )
                        ts_ = g.config.get('animation_timestamp', {})
                        if ts_ and str2bool(ts_.get('enabled')):
                            img = timestamp_it(img, ts_, ts_h=h, ts_w=w)
                        images.append(img)
                        all_grabbed_frames.append(i)
                    except Exception as e:
                        g.logger.error(
                            f"{log_prefix} error during image frame grab (includes resize and timestamp): {e}"
                        )
                end_grabbing_frames = datetime.datetime.now() - start_grabbing_frames
                g.logger.debug(
                    2,
                    f"{log_prefix} grabbed {len(all_grabbed_frames)} frames in "
                    f"{round(end_grabbing_frames.total_seconds(), 3)} sec Frame Ids: {all_grabbed_frames}",
                )

            if ani_type == "mp4":
                od_images = []
                g.logger.debug(f"{log_prefix} MP4 requested...")
                if image is not None:
                    for i in range(4):
                        od_images.append(image)
                od_images.extend(images)
                imageio.mimwrite(f"{file_name}.mp4", od_images, format="mp4", fps=target_fps)
                mp4_file = Path(f"{file_name}.mp4")
                size = mp4_file.stat().st_size
                g.logger.debug(
                    f"{log_prefix} saved to {mp4_file.name}, size {size / 1024 / 1024:.2f} MB, frames: {len(images)}"
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
                    f"{log_prefix} {'fast ' if fast_gif else 'regular speed '}GIF requested...",
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
                    # g.logger.debug(f"{gif_buffer_seconds=} | {target_fps=} | {gif_start_frame=} | {gif_end_frame=} | sliced from {s1=} | negative {s2=}")
                    g.logger.debug(
                        f"{log_prefix}{'fast ' if fast_gif is not None else ''}gif: sliced {s1} to"
                        f" -{s2} from a total of {len(images)}, writing to disk..."
                    )
                    g.logger.debug(
                        f"{log_prefix}{'fast ' if fast_gif is not None else ''}gif: optimizing GIF using gifsicle"
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
                        f"perf:{log_prefix}{'fast ' if fast_gif is not None else ''}gif: {diff_write} sec to optimize "
                        f"and save {ani_type} to disk -> before: {before_opt_size / 1024 / 1024:.2f} MB --> "
                        f"after optimization: {size / 1024 / 1024:.2f} MB for {len(gif_images)} frames"
                    )
                else:
                    g.logger.debug(
                        f"{log_prefix}{'fast ' if fast_gif is not None else ''}gif: range is weird start: s1='{s1}' "
                        f"end offset: s2='-{s2}'"
                    )
    g.animation_seconds = time.perf_counter()-perf


def resize_image(img: cv2, resize_w, quiet=None):
    lp = "resize:img:"
    if resize_w == 'no':
        g.logger.debug(f"{lp} 'resize' is set to 'no', not resizing image...") if not quiet else None
    elif img is not None:
        (h, w) = img.shape[:2]
        try:
            resize_w = float(resize_w)
        except Exception as all_ex:
            g.logger.error(
                f"{lp} 'resize' must be set to 'no' or a number like 800 or 320.55, any "
                f"other format will cause errors (currently set to {resize_w}), not resizing image..."
            ) if not quiet else None
        else:
            asp = float(resize_w) / float(w)
            dim = (int(resize_w), int(h * asp))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            g.logger.debug(2, f"{lp} success using resize={resize_w} - original dimensions: {w}*{h}"
                              f" - resized dimensions: {dim[1]}*{dim[0]}"
                           ) if not quiet else None
    else:
        g.logger.debug(f"{lp} 'resize' called but no image supplied!") if not quiet else None
    return img


class my_stderr:
    def __init__(self):
        self.tot_errmsg = []

    def write(self, data, *args, **kwargs):
        self.tot_errmsg.append(data)

    def flush(self, *args, **kwargs):
        idx = min(len(stack()), 1)
        caller = getframeinfo(stack()[idx][0])
        g.logger.error(
            f"'std.err' --> {' '.join(self.tot_errmsg).rstrip()}", caller=caller
        ) if len(self.tot_errmsg) else None
        self.tot_errmsg = []
        sys.__stderr__.flush()

    @staticmethod
    def close(*args, **kwargs):
        g.logger.debug(f"STDERR CLOSE -> {args if args else None} {kwargs if kwargs else None}")
        sys.__stderr__.close()


class my_stdout:
    def __init__(self):
        self.tot_msg = []

    @staticmethod
    def write(data, *args, **kwargs):
        idx = min(len(stack()), 1)
        caller = getframeinfo(stack()[idx][0])
        if data != "\n":
            g.logger.debug(f"'std.out' --> {data}", caller=caller)

    def flush(self, *args, **kwargs):
        idx = min(len(stack()), 1)
        caller = getframeinfo(stack()[idx][0])
        g.logger.error(
            f"'std.out' FLUSH --> {' '.join(self.tot_msg).rstrip()}", caller=caller
        ) if len(self.tot_msg) else None
        self.tot_msg = []
        sys.__stdout__.flush()

    @staticmethod
    def close(*args, **kwargs):
        sys.__stdout__.close()
        g.logger.debug(f"STDOUT CLOSE -> {args if args else None} {kwargs if kwargs else None}")


def pop_coco_names(file_name, globs):
    global g
    g = globs
    ret_val = []
    lp = 'coco names:'
    if Path(file_name).exists() and Path(file_name).is_file():
        g.logger.debug(f"{lp} attempting to populate COCO names using file: '{file_name}'")
        coco = open(file_name, 'r')
        for line in coco:
            line = str(line).replace('\n', '')
            ret_val.append(line)
        coco.close()
        g.logger.debug(f"{lp} successfully populated {len(ret_val)} COCO labels from '{coco.name}'")
    elif not Path(file_name).exists():
        pass
    elif not Path(file_name).is_file():
        pass
    return ret_val


def do_hass(hass_globals):
    from urllib3.exceptions import InsecureRequestWarning, NewConnectionError
    from urllib3 import disable_warnings
    import requests
    # turn off insecure warnings for self signed certificates
    disable_warnings(InsecureRequestWarning)

    g = hass_globals
    log_prefix = "hass add-on:"
    headers = {
        "Authorization": f"Bearer {g.config.get('hass_token')}",
        "content-type": "application/json",
    }
    sensor = g.config.get("hass_notify")
    cooldown = g.config.get("hass_cooldown")
    ha_url = f"{g.config.get('hass_server')}/api/states/"

    # TODO: add person.<entity> logic
    resp = None
    # First check if HA is not set up and use the local backup if configured
    if not sensor and not cooldown:
        g.logger.debug(
            4,
            f"{log_prefix} You have HomeAssistant API support for pushover enabled but have not setup any"
            f" sensors to control the sending of pushover notifications. "
            f"Set global and/or per monitor sensors to control them. Checking for local config option "
            f"'push_cooldown'",
        )

        send_push = True
        # print(f"{g.config=}")
        # check if push_cooldown is set
        if g.config.get("push_cooldown"):
            g.logger.debug(
                f"{log_prefix} no homeassistant sensors configured, "
                f"using 'push_cooldown' -> {g.config.get('push_cooldown')}"
            )
            try:
                cooldown = float(g.config.get("push_cooldown"))
            except TypeError as ex:
                g.logger.error(
                    f"{log_prefix} 'push_cooldown' malformed, sending push..."
                )
            else:
                time_since_last_push = pkl_pushover(
                    "load", mid=g.mid
                )
                if time_since_last_push:
                    now = datetime.datetime.now()
                    differ = (
                            now - time_since_last_push
                    ).total_seconds()
                    if differ < cooldown:
                        g.logger.debug(
                            f"{log_prefix} COOLDOWN elapsed-> {differ} / {cooldown} "
                            f"skipping notification..."
                        )
                        send_push = False
                    else:
                        g.logger.debug(
                            f"{log_prefix} COOLDOWN elapsed-> {differ} / {cooldown} "
                            f"sending notification..."
                        )
                    cooldown = None
    # connect to HASS for data on the helpers
    elif sensor:
        # Toggle Helper aka On/Off
        ha_sensor_url = ha_url + sensor
        # todo attempt loop for hass -> better error handling for seem less pushover add-on
        try:
            resp = requests.get(
                ha_sensor_url, headers=headers, verify=False
            ).json()  # strict cert checking off, encryption still works.
        except NewConnectionError as n_ex:
            g.logger.error(
                f"{log_prefix} failed to make a new connection to the HASS host '{ha_url}', "
                f"sending push")
            send_push = True
        except Exception as ex:
            g.logger.error(
                f"{log_prefix}err_msg-> {ex}"
            )
            g.logger.debug(
                f"traceback -> {format_exc()}"
            )
            send_push = True
        else:
            if resp.get('message') == 'Entity not found.':
                g.logger.error(
                    f"{log_prefix} the configured sensor -> '{sensor}' can not be found on the HASS host!"
                    f" check for spelling or formatting errors!"
                )
                send_push = True
            else:
                g.logger.debug(
                    f"{log_prefix} the Toggle Helper sensor for monitor {g.mid} has returned -> '{resp.get('state')}'")
                # The sensor returns on or off, str2bool converts that to True/False Boolean
                send_push = str2bool(resp.get("state"))
    else:
        send_push = True

    if cooldown and (
            (
                    sensor
                    and (resp is not None and str2bool(resp.get("state")))
            )
            or (not sensor)
    ):
        try:
            ha_cooldown_url = f"{ha_url}{cooldown}"
            cooldown_response = requests.get(
                ha_cooldown_url, headers=headers
            )
        except Exception as ex:
            g.logger.error(
                f"{log_prefix}err_msg-> {ex}"
            )
            send_push = True
        else:
            resp = cooldown_response.json()
            int_val = float(resp.get("state", 1))
            g.logger.debug(
                f"{log_prefix} the Number Helper (cool down) sensor for monitor {g.mid} has returned -> "
                f"'{resp.get('state')}'"
            )
            time_since_last_push = pkl_pushover("load", mid=g.mid)
            if time_since_last_push:

                differ = (datetime.datetime.now() - time_since_last_push).total_seconds()
                if differ < int_val:
                    g.logger.debug(
                        f"{log_prefix} SKIPPING NOTIFICATION -> elapsed: {differ} "
                        f"- maximum: {int_val}"
                    )
                    send_push = False
                else:
                    g.logger.debug(
                        f"{log_prefix} seconds elapsed since last successful live event "
                        f"pushover notification -> {differ} - maximum: {int_val}, allowing notification"
                    )
                    send_push = True
            else:
                send_push = True
    else:  # HASS Toggle Helper for On/Off and local 'push_cooldown' for cooldown
        if g.config.get("push_cooldown"):
            g.logger.debug(
                f"{log_prefix} there is no homeassistant integration configured for cooldown, "
                f"using config 'push_cooldown' -> {g.config.get('push_cooldown')}"
            )
            try:
                cooldown = float(g.config.get("push_cooldown"))
            except Exception as ex:
                g.logger.error(
                    f"{log_prefix} 'push_cooldown' malformed, sending push..."
                )
                send_push = True
            else:
                time_since_last_push = pkl_pushover("load", mid=g.mid)
                if time_since_last_push:
                    differ = (
                            datetime.datetime.now() - time_since_last_push
                    ).total_seconds()
                    if differ < cooldown:
                        g.logger.debug(
                            f"{log_prefix} COOLDOWN elapsed-> {differ} / {cooldown} skipping notification..."
                        )
                        send_push = False
                    else:
                        g.logger.debug(
                            f"{log_prefix} COOLDOWN elapsed-> {differ} / {cooldown} sending notification..."
                        )
                        send_push = True
                else:
                    send_push = True
        else:
            send_push = True
    return send_push


def id_generator(size=16, chars=ascii_letters + digits) -> str:
    return "".join(choice(chars) for _ in range(size))


def digit_generator(size=16, digits_=digits) -> str:
    return "".join(choice(digits_) for _ in range(size))


def de_dup(task, separator=None, return_str=False) -> list:
    """Removes duplicates in a string or list, if string you can also pass a separator (default: ',').
    :param return_str: (bool) - return a space seperated string instead of a list
    :param task: (str) or (list) - strings or list of strings that you want duplicates removed from
    :param separator: (str) - seperator for task if its a str
    :returns: list of de-duplicated strings"""
    if separator is None:
        separator = ","
    ret_list = []
    append = None
    if isinstance(task, str):
        snapshots = 0
        for x in task.split(separator):
            if str(x).startswith("s") and snapshots < 1:
                snapshots += 1
                append = True
            elif x not in ret_list:
                append = True
            if append:
                # g.logger.debug(f"DEDUP STR -> appending {x} ")
                ret_list.append(x)
                append = None
        # [ret_list.append(x) for x in task.split(seperator) if x not in ret_list]
    elif isinstance(task, list):
        snapshots = 0
        for x in task:
            if str(x).startswith("s") and snapshots < 1:
                snapshots += 1
                append = True
            elif x not in ret_list:
                append = True
            if append:
                ret_list.append(x)
                append = None

        # [ret_list.append(x) for x in task if x not in ret_list]

    return ret_list if not return_str else " ".join([str(x) for x in ret_list])


def read_config(file: str, return_object=False) -> dict or ConfigParser:
    """Returns a ConfigParser object or a dict of the file without sections split up (doesnt decode and replace
    secrets though)
    """
    config_file: ConfigParser = ConfigParser(interpolation=None, inline_comment_prefixes="#")
    with open(file) as f:
        config_file.read_file(f)
    if return_object:
        return config_file  # return whole ConfigParser object if requested
    config_file.optionxform = str  # converts to lowercase strings, so MQTT_PASSWORD is now mqtt_password, etc.
    return config_file._sections  # return a dict object that removes sections and is strictly { option: value }


def write_text(
        frame=None,
        text=None,
        text_color: tuple = (0, 0, 0),
        x=None,
        y=None,
        w=None,
        h=None,
        adjust: bool = False,
        font: cv2 = None,
        font_scale: float = None,
        thickness: int = 1,
        bg: bool = True,
        bg_color: tuple = (255, 255, 255),
):
    if frame is None:
        g.logger.error(f"write text: called without supplying an image")
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
    lp = "image:write text:"
    if adjust:
        if not w or not h:
            # TODO make it enlarge also if too small
            g.logger.error(f"{lp} cannot auto adjust text as "
                           f"{'W ' if not w else ''}{'and ' if not w and not h else ''}{'H ' if not h else ''}"
                           f"not provided")
        else:
            if x + tw > w:
                print(f"adjust needed, text would go out of frame width")
                x = max(0, x - (x + tw - w))

            if y + th > h:
                print(f"adjust needed, text would go out of frame height")
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
    # print(
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
        image=None,
        boxes=None,
        labels=None,
        confidences=None,
        polygons=None,
        box_color=None,
        poly_color=(255, 255, 255),
        poly_thickness=1,
        write_conf=True,
        errors=None,
        write_model=False,
        models=None,
):
    # FIXME: need to add scaling dependant on image dimensions
    # print (1,"**************DRAW BBOX={} LAB={}".format(boxes,labels))
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
    slate_colors = [
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
            # if models[i] == "face_dlib":
            #     smodel = "Fd"
            # elif models[i] == "face_tpu":
            #     smodel = "Ft"
            # elif models[i] == "yolo[gpu]":
            #     smodel = "Yg"
            # elif models[i] == "yolo[cpu]":
            #     smodel = "Yc"
            # elif models[i] == 'alpr':
            #     smodel = 'ALPR'
            # elif models[i] == 'coral':
            #     smodel = 'Cor'
            # else:
            #     smodel = models[i][0].upper()  # C for coral
            # label += f"[{smodel}]"
            label += f"[{models[i]}]"
        # draw bounding box around object
        # g.logger.debug(f"{lp} {boxes=} -------- {polygons=}")
        # print (f"{lp} DRAWING COLOR={box_color} RECT={boxes[i][0]},{boxes[i][1]} {boxes[i][2]},{boxes[i][3]}")
        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), box_color, 2)

        # write text
        font_thickness = 1
        font_scale = 0.6
        # todo: scaling for different resolutions
        # add something better than this
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
        # print(
        #     f"DRAW BBOX - WRITE TEXT {h=} -- {w=}   {font_scale=} -- {font_thickness=} -- text_width={text_size[0]} -- text_height={text_size[1]}")

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
    return [tuple(map(int, x.strip().split(","))) for x in string.split(" ")]


def str2arr(string: str, delimiter=",") -> list:
    """send a comma delimited"""
    return [map(int, x.strip().split(",")) for x in string.split(" ")]


def str_split(my_str):
    return [x.strip() for x in my_str.split(",")]


def str2bool(v: Optional[Union[str, bool]]) -> Optional[Union[str, bool]]:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    v = str(v)
    true_ret = ("yes", "true", "t", "y", "1", "on", "ok", "okay")
    false_ret = ("no", "false", "f", "n", "0", "off")
    if v.lower() in true_ret:
        return True
    elif v.lower() in false_ret:
        return False
    else:
        return g.logger.error(
            f"str2bool: '{v}' is not able to be parsed into a boolean operator"
        )


def verify_vals(config: dict, vals: set) -> bool:
    """Verify that the list of strings in *vals* is contained within the dict of *config*.
            **Args:**
                - *config* (dict): containing all config values.
                - *vals* (set of str): containing strings of the name of the keys you want to match in the config
                dictionary.

    Returns: *bool*
    """
    ret = []
    for val in vals:
        if val in config:
            ret.append(val)
    return True if len(ret) == len(vals) else False


def import_zm_zones(reason, conf_globals, existing_polygons):
    if existing_polygons is None or not existing_polygons:
        existing_polygons = []
    g = conf_globals
    match_reason = False
    lp = "import zm zones:"
    if reason:
        match_reason = str2bool(g.config["only_triggered_zm_zones"])
    g.logger.debug(2, f"{lp} only trigger on ZM zones: {match_reason}  reason for event: {reason}")
    url = f"{g.config['portal']}/api/zones/forMonitor/{g.mid}.json"
    j = g.api.make_request(url)
    # Now lets look at reason to see if we need to honor ZM motion zones
    for zone_ in j["zones"]:
        # print(f"{lp} ********* ITEM TYPE {zone_['Zone']['Type']}")
        if str(zone_["Zone"]["Type"]).lower == 'inactive':
            g.logger.debug(
                2,
                f"{lp} skipping '{zone_['Zone']['Name']}' as it is set to 'Inactive' in Zoneminder"
            )
            continue
        if match_reason:
            if not findWholeWord(zone_["Zone"]["Name"])(reason):
                g.logger.debug(
                    f"{lp}:triggered by ZM: not importing '{zone_['Zone']['Name']}' as it is not in event alarm cause"
                    f" -> '{reason}'"
                )
                continue
        g.logger.debug(2, f"{lp} '{zone_['Zone']['Name']}' @ [{zone_['Zone']['Coords']}] is being added to polygons")
        existing_polygons.append(
            {
                "name": zone_["Zone"]["Name"].replace(" ", "_").lower(),
                "value": str2tuple(zone_["Zone"]["Coords"]),
                "pattern": None,
            }
        )
    return existing_polygons


def pkl_pushover(action: str = "load", time_since_sent=None, mid=None):
    from pickle import load as pickle_load, dump as pickle_dump

    pkl_path = (
            f"{g.config.get('base_data_path')}/push" or "/var/lib/zmeventnotification/push"
    )
    mon_file = f"{pkl_path}/mon-{mid}-pushover.pkl"
    if action == "load":
        g.logger.debug(2, f"push:pkl:trying to load '{mon_file}'")
        try:
            with open(mon_file, "rb") as fh:
                time_since_sent = pickle_load(fh)
            return time_since_sent
        except FileNotFoundError:
            g.logger.debug(
                f"push:pkl:FileNotFound - no time of last successful push found for monitor {mid}",
            )
            return
        except EOFError:
            g.logger.debug(
                f"push:pkl:empty file found for monitor {mid}, going to remove '{mon_file}'",
            )
            try:
                os.remove(mon_file)
            except Exception as e:
                g.logger.error(f"push:pkl:could not delete: {e}")
        except Exception as e:
            g.logger.error(f"push:pkl:error Exception = {e}")
            g.logger.error(f"Traceback:{format_exc()}")
    elif action == "write":
        try:
            with open(mon_file, "wb") as fd:
                pickle_dump(time_since_sent, fd)
                g.logger.debug(
                    4, f"push:pkl:time since sent:{time_since_sent} to '{mon_file}'"
                )
        except Exception as e:
            g.logger.error(
                f"push:pkl:error writing to '{mon_file}', time since last successful push sent not recorded:{e}"
            )


def get_image(path, cause):
    prefix = None
    if cause.startswith("["):
        prefix = str(cause).split("]")[0]
        prefix = prefix.strip("[")
    if os.path.exists(f"{path}/objdetect.gif"):
        return f"{path}/objdetect.gif"
    if os.path.exists(f"{path}/objdetect.jpg"):
        return f"{path}/objdetect.jpg"
    if prefix == "a":
        return f"{path}/alarm.jpg"
    return f"{path}/snapshot.jpg"


# credit: https://stackoverflow.com/a/5320179
def findWholeWord(w):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


def grab_frameid(frameid_str: str) -> str:
    # removes the s- or a- from frame ID string
    if len(str(frameid_str).split("-")) > 1:
        frameid = frameid_str.split("-")[1]
    else:
        frameid = frameid_str
    return frameid


def pkl(action, boxes=None, labels=None, confs=None, event=None):
    if boxes is None:
        boxes = []
    if labels is None:
        labels = []
    if confs is None:
        confs = []
    if event is None:
        event = ""
    saved_bs, saved_ls, saved_cs, saved_event = None, None, None, None
    image_path = (
            f"{g.config.get('base_data_path')}/images"
            or "/var/lib/zmeventnotification/images"
    )
    mon_file = f"{image_path}/monitor-{g.mid}-data.pkl"
    if action == "load":
        g.logger.debug(2, f"pkl: trying to load file: '{mon_file}'")
        try:
            with open(mon_file, "rb") as fh:
                saved_bs = pickle_load(fh)
                saved_ls = pickle_load(fh)
                saved_cs = pickle_load(fh)
                saved_event = pickle_load(fh)
        except FileNotFoundError:
            g.logger.debug(f"pkl: no history data file found for monitor '{g.mid}'")
        except EOFError:
            g.logger.debug(f"pkl: empty file found for monitor '{g.mid}'")
            g.logger.debug(f"pkl: going to remove '{mon_file}'")
            try:
                os.remove(mon_file)
            except Exception as e:
                g.logger.error(f"pkl: could not delete: {e}")
        except Exception as e:
            g.logger.error(f"pkl: error: {e}")
            g.logger.error(f"pkl:traceback: {format_exc()}")
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
                    f"pkl: saved_event:{event} saved boxes:{boxes} - labels:{labels} - confs:{confs} to file: '{mon_file}'",
                )
        except Exception as e:
            g.logger.error(
                f"pkl: error writing to '{mon_file}' past detections not recorded, err msg -> {e}"
            )


def get_www_user():
    """:returns: ('webuser','webgroup')"""
    import pwd
    import grp

    webuser = []
    webgrp = []
    try:
        u_apache = pwd.getpwnam("apache")
    except Exception:
        pass
    else:
        webuser = ["apache"]
    try:
        u_www = pwd.getpwnam("www-data")
    except Exception:
        pass
    else:
        webuser = ["www-data"]
    try:
        g_apache = grp.getgrnam("apache")
    except Exception:
        pass
    else:
        webgrp = ["apache"]
    try:
        g_apache = grp.getgrnam("www-data")
    except Exception:
        pass
    else:
        webgrp = ["www-data"]

    return "".join(webuser), "".join(webgrp)



class Pushover:
    def __init__(self):
        """Create a PushOver object to send pushover notifications via `request.post` to API"""
        self.push_send_url = "https://api.pushover.net/1/messages.json"
        self.options = None
        self.file = None

    def check_config(self, config: dict) -> bool:
        req = ("user", "token")
        tot_reqs = 0
        for k, v in config.items():
            if k in req:
                tot_reqs += 1
        if tot_reqs == len(req):
            return True
        else:
            return False

    def send(self, param_dict, files=None, record_last=True):
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

                    'priority': 0,

                    'device': 'a specific device',

                        }

        """
        from requests import post

        self.options = param_dict

        self.file = files if files else self.file
        if not self.check_config(self.options):
            g.logger.error(
                f"you must specify at a minimum a push_key"
                f" and push_token in the config file to send pushover notification data"
            )
            return
        try:
            r = post(self.push_send_url, data=self.options, files=self.file)
            r.raise_for_status()
            r = r.json()
        except Exception as ex:
            g.logger.error(
                f"pushover: sending notification data and converting response to JSON FAILED -> {ex}"
            )
        else:
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
                        # if str(md_val).startswith('<np.as'): return
                        first_tight_line += 1
                        xb = True if first_tight_line == 1 else None
                        # print(f"dval is dict ... {first_tight_line=} sending nl={xb}")
                        if i_idx == 0:
                            g.logger.debug("--- --- ---")
                        g.logger.debug(
                            f"'{dkey}'->  {md_key}-->{md_val}  ",
                            tight=True,
                            nl=xb,
                        )
            elif isinstance(dval, list):
                if dval.__len__() > 0:
                    # for i_idx, (md_key, md_val) in enumerate(dval[0].items()):
                    for all_match in dval:
                        for i_idx, (md_key, md_val) in enumerate(all_match.items()):
                            if not md_val:
                                continue
                            if len(str(md_val)):
                                if i_idx == 0:
                                    g.logger.debug("--- --- ---")
                                first_tight_line += 1
                                xb = True if first_tight_line == 1 else None
                                # print(f"dval is list and it has a dict inside ... {first_tight_line=}")
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
                    # print(f"dval is list without a dict ... {first_tight_line=}")
                    g.logger.debug(
                        f"{dkey}->  {dval}  ", tight=True, nl=xb
                    )
            else:
                first_tight_line += 1
                xb = True if first_tight_line == 1 else None
                # print(f"dval is NOT list OR dict ... {first_tight_line=}")
                g.logger.debug(f"'{dkey}'->  {dval}  ", tight=True, nl=xb)


class LogBuffer:
    @staticmethod
    def kwarg_parse(**kwargs):
        caller, level, debug_level, message = None, 'DBG', 1, None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'level':
                level = v
            elif k == 'message':
                message = v
            elif k == 'debug_level':
                debug_level = v
        return {'message': message, 'caller': caller, 'level': level, 'debug_level': debug_level}

    def __init__(self):
        self.buffer: Optional[list] = []

    # make it iterable
    def __iter__(self):
        if self.buffer:
            for _line in self.buffer:
                yield _line

    def pop(self):
        if self.buffer:
            return self.buffer.pop()

    # return length of buffer
    def __len__(self):
        if self.buffer:
            return len(self.buffer)

    def store(self, **kwargs):
        caller, level, debug_level, message = None, 'DBG', 1, None
        kwargs = self.kwarg_parse(**kwargs)
        dt = time_format(datetime.datetime.now())
        # print (len(stack()))
        idx = min(len(stack()), 2)  # in the case of someone calls this directly
        caller = getframeinfo(stack()[idx][0])
        message = kwargs['message']
        # print ('CALLER INFO --> FILE: {} LINE: {}'.format(caller.filename, caller.lineno))
        disp_level = level
        if level == 'DBG':
            disp_level = f'DBG{debug_level}'
        data_structure = {
            'timestamp': dt,
            'display_level': disp_level,
            'filename': Path(caller.filename).name,
            'lineno': caller.lineno,
            'message': message,
        }
        self.buffer.append(data_structure)

    def info(self, message, *args, **kwargs):
        level = 'INF'
        if message is not None:
            self.store(
                level=level,
                message=message,
            )

    def debug(self, *args, **kwargs):
        level = 'DBG'
        debug_level = 1
        message = None
        if len(args) == 1:
            level = 1
            message = args[0]
        elif len(args) == 2:
            level = args[0]
            message = args[1]
        if message is not None:
            # self.buffer.append(message)
            self.store(
                level=level,
                debug_level=debug_level,
                message=message
            )

    def warning(self, message, *args, **kwargs):
        level = 'WAR'
        if message is not None:
            # self.buffer.append(message)
            self.store(
                level=level,
                message=message
            )

    def error(self, message, *args, **kwargs):
        level = 'ERR'
        if message is not None:
            # self.buffer.append(message)
            self.store(
                level=level,
                message=message
            )

    def fatal(self, message, *args, **kwargs):
        level = 'FAT'
        if message is not None:
            # self.buffer.append(message)
            self.store(
                level=level,
                message=message
            )
        self.log_close(exit=-1)

    def panic(self, message, *args, **kwargs):
        level = 'PNC'
        if message is not None:
            # self.buffer.append(message)
            self.store(
                level=level,
                message=message
            )
        self.log_close(exit=-1)

    def log_close(self, *args, **kwargs):
        if self.buffer and len(self.buffer):
            # sort it by timestamp
            self.buffer = sorted(self.buffer, key=lambda x: x['timestamp'], reverse=True)
            for _ in range(len(self.buffer)):
                line = self.buffer.pop()
                if line:
                    fnfl = f"{line['filename']}:{line['lineno']}"
                    print_log_string = (f"{line['timestamp']} LOG_BUFFER[{os.getpid()}] {line['display_level']} " 
                                       f"{fnfl}->[{line['message']}]")
                    print(print_log_string)
        if kwargs.get('exit') is not None:
            exit(kwargs['exit'])
        return


def time_format(dt_form):
    if len(str(float(f"{dt_form.microsecond / 1e6}")).split(".")) > 1:
        micro_sec = str(float(f"{dt_form.microsecond / 1e6}")).split(".")[1]
    else:
        micro_sec = str(float(f"{dt_form.microsecond / 1e6}")).split(".")[0]
    # pad the microseconds with zeros
    while len(micro_sec) < 6:
        micro_sec = f"0{micro_sec}"
    return f"{dt_form.strftime('%m/%d/%y %H:%M:%S')}.{micro_sec}"


def do_mqtt(args, et, pred, pred_out, notes_zone, matched_data, push_image, globs):
    from pyzm.helpers.mqtt import Mqtt
    log_prefix = "mqtt add-on:"
    try:
        mqtt_topic = g.config.get("mqtt", {}).get("mqtt_topic", "zmes")
        g.logger.debug(f"{log_prefix} is enabled, initialising...")
        mqtt_conf = {
            "mqtt_enable": g.config.get("mqtt_enable"),
            "mqtt_force": g.config.get("mqtt_force"),
            "mqtt_broker": g.config.get("mqtt_broker"),
            "mqtt_user": g.config.get("mqtt_user"),
            "mqtt_pass": g.config.get("mqtt_pass"),
            "mqtt_port": g.config.get("mqtt_port"),
            "mqtt_topic": g.config.get("mqtt_topic"),
            "mqtt_retain": g.config.get("mqtt_retain"),
            "mqtt_qos": g.config.get("mqtt_qos"),
            "mqtt_tls_allow_self_signed": g.config.get("mqtt_tls_allow_self_signed"),
            "mqtt_tls_insecure": g.config.get("mqtt_tls_insecure"),
            "tls_ca": g.config.get("mqtt_tls_ca"),
            "tls_cert": g.config.get("mqtt_tls_cert"),
            "tls_key": g.config.get("mqtt_tls_key"),
        }
        mqtt_obj = Mqtt(config=mqtt_conf, globs=globs)
        mqtt_obj.connect()
    except Exception as e:
        g.logger.error(f"{log_prefix} constructing err_msg-> {e}")
        print(format_exc())
    else:
        if not args.get('file'):
            mqtt_obj.create_ml_image(args.get("eventpath"), pred)
            mqtt_obj.publish(topic=f"{mqtt_topic}/picture/{g.mid}", retain=g.config.get("mqtt_retain"))
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
                    "labels": matched_data.get('labels'),
                    "conf": matched_data.get('confidences'),
                    "bbox": matched_data.get('boxes'),
                    "models": matched_data.get('model_names'),
                }
            )
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/rdata/{g.mid}", message=det_data, retain=g.config.get("mqtt_retain")
            )

        else:
            # convert image to a byte array
            # cv2.imencode('.jpg', frame)[1].tobytes()
            # push_image = cv2.cvtColor(push_image, cv2.COLOR_BGR2RGB)
            push_image = cv2.imencode('.jpg', push_image)[1].tobytes()
            mqtt_obj.publish(topic=f"{mqtt_topic}/picture/file", message=push_image, retain=g.config.get("mqtt_retain"),)
            # build this with info for the FILE
            detection_info = json.dumps(
                {
                    "file_name": args.get("file"),
                    "labels": matched_data.get('labels'),
                    "conf": matched_data.get('confidences'),
                    "bbox": matched_data.get('boxes'),
                    "models": matched_data.get('model_names'),
                    "detection_type": matched_data.get('type'),
                }
            )
            mqtt_obj.publish(
                topic=f"{mqtt_topic}/data/file", message=detection_info, retain=g.config.get("mqtt_retain"),
            )
        mqtt_obj.close()



def mlapi_import_zones(conf_globals=None, config_obj=None):
    g = conf_globals
    lp = "mlapi:import zm zones:"
    zones = g.api.zones()
    c = config_obj
    if zones:
        for zone in zones:
            type_ = str(zone.type()).lower()
            mid = zone.monitorid()
            name = zone.name()
            coords = zone.coords()
            if type_ == 'inactive':
                g.logger.debug(
                    f"{lp} skipping {name} as it is not a zone which we are expecting activity, "
                    f"type: {type_}"
                )
                continue

            if mid not in c.polygons:
                c.polygons[mid] = []

            name = name.replace(' ', '_').lower()
            g.logger.debug(2,
                           f"{lp} IMPORTING '{name}' @ [{coords}] from monitor '{mid}'")
            c.polygons[mid].append({
                'name': name,
                'value': str2tuple(coords),
                'pattern': None
            })
        # iterate polygons and apply matching detection patterns by zone name
        for poly in c.polygons[mid]:
            if poly["name"] in c.detection_patterns:
                poly["pattern"] = c.detection_patterns[poly['name']]
                g.logger.debug(
                    2,
                    f"{lp} overriding match pattern for zone/polygon '{poly['name']}' with: "
                    f"{c.detection_patterns[poly['name']]}"
                )
    return c


