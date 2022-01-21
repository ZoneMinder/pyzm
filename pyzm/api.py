"""
ZMApi
=============
Python API wrapper for ZM.
Exposes login, monitors, events, etc. API

Important:

  Make sure you have the following settings in ZM:

  - ``AUTH_RELAY`` is set to hashed
  - A valid ``AUTH_HASH_SECRET`` is provided (not empty)
  - ``AUTH_HASH_IPS`` is disabled
  - ``OPT_USE_APIS`` is enabled
  - If you are using any version lower than ZM 1.34, ``OPT_USE_GOOG_RECAPTCHA`` is disabled
  - If you are NOT using authentication at all in ZM, that is ``OPT_USE_AUTH`` is disabled, then make sure you
  also disable authentication in zmNinja, otherwise it will keep waiting for auth keys.
  - I don't quite know why, but on some devices, connection issues are caused because ZoneMinder's CSRF code
   causes issues. See `this <https://forums.zoneminder.com/viewtopic.php?f=33&p=115422#p115422>`__ thread, for
   example. In this case, try turning off CSRF checks by going to  ``ZM->Options->System`` and disable
   "Enable CSRF magic".

"""

import datetime

import requests
from requests import Session
from requests.packages.urllib3 import disable_warnings
from requests.exceptions import HTTPError
from urllib3.exceptions import InsecureRequestWarning

from inspect import getframeinfo, stack
from typing import Optional, Dict, List, AnyStr

from pyzm.helpers.Configs import Configs
from pyzm.helpers.Events import Events
from pyzm.helpers.Monitors import Monitors
from pyzm.helpers.States import States
from pyzm.helpers.Zones import Zones
from pyzm.interface import GlobalConfig

g: GlobalConfig
GRACE = 60 * 5  # 5 mins
lp = 'api:'


class ZMApi:
    def __init__(self, options=None, kickstart=None):
        """
        options (dict):
            - apiurl (str) - the full API URL (example https://server/zm/api)
            - portalurl (str) - the full portal URL (example https://server/zm). Only needed if you are downloading events/images
            - user (str) - username (None if using 'no auth')
            - password (str) - password (None if using 'no auth')
            - disable_ssl_cert_check (bool) - if True will let you use self-signed certs
            - basic_auth_user (str) - basic auth username
            - basic_auth_password (str) - basic auth password




        kickstart - (dict) containing existing JWT token data supplied to MLAPI by ZMES (saves time by skipping login).
              - user (str): Username,
              - password (str): Password,
              - access_token (str): Access token,
              - refresh_token (str): Refresh token,
              - access_token_expires (str|int): Access token expiration time in seconds (Example: 3600),
              - refresh_token_expires (str|int): Refresh token expiration time in seconds (Example: 3600),
              - access_token_datetime (str|float): Access token datetime in timestamp format,
              - refresh_token_datetime (str|float): Refresh token datetime in timestamp format,
              - api_version (str): API version,
              - zm_version (str): ZoneMinder version,
        """
        global g
        g = GlobalConfig()
        idx = min(len(stack()), 1)  # in case someone calls this directly
        caller = getframeinfo(stack()[idx][0])
        if options is None:
            options = {}
        # print(f"{options = }")
        self.api_url = options.get("apiurl")
        self.portal_url = options.get("portalurl")
        if not self.portal_url and (
                self.api_url
                and isinstance(self.api_url, str)
                and self.api_url.endswith("/api")
        ):
            self.portal_url = self.api_url[: -len("/api")]
            g.logger.debug(
                2,
                f"{lp} portal not passed, guessing portal URL from portal_api is: {self.portal_url}", caller=caller)

        self.options = options

        self.sanitize = False
        self.auth_type: Optional[str] = None
        self.authenticated = False
        self.auth_enabled = True
        self.access_token = ""
        self.refresh_token = ""
        self.access_token_expires = None
        self.refresh_token_expires = None
        self.refresh_token_datetime = None
        self.access_token_datetime = None
        self.legacy_credentials = None
        self.api_version = ''
        self.zm_version = ''
        self.zm_tz = None

        self.Monitors = None
        self.Events = None
        self.Configs = None
        self.Zones = None
        self.States: Optional[List[States]] = None
        self.session = Session()
        if self.options.get("sanitize_portal"):
            self.sanitize = True
        if self.options.get(
                "disable_ssl_cert_check", True
        ):  # this turns off certificate verification
            self.session.verify = False
            g.logger.debug(
                2,
                f"{lp} SSL certificate verification disabled (encryption enabled, vulnerable to MITM attacks)",
                caller=caller,
            )
            disable_warnings(category=InsecureRequestWarning)
        if kickstart:
            self.options['user'] = kickstart.get('user')
            self.options['password'] = kickstart.get('password')
            self.auth_type = "token"
            self.access_token = kickstart.get("access_token")
            self.refresh_token = kickstart.get("refresh_token")
            self.access_token_expires = kickstart.get("access_token_expires")
            self.refresh_token_expires = kickstart.get("refresh_token_expires")
            self.api_version = kickstart.get("api_version")
            self.zm_version = kickstart.get("zm_version")
            self.authenticated = True
            self.refresh_token_datetime = datetime.datetime.fromtimestamp(
                float(kickstart.get("refresh_token_datetime"))
            )
            self.access_token_datetime = datetime.datetime.fromtimestamp(
                float(kickstart.get("access_token_datetime"))
            )
            g.logger.debug(
                2,
                f"{lp}KICKSTART: an AUTH JWT and associated data has been supplied, no need to re login to ZM API",
                caller=caller,
            )
        else:
            if self.options.get("basic_auth_user"):
                print(f"{self.options.get('basic_auth_user') = }")
                g.logger.debug(4, f"{lp} basic auth requested, configuring", caller=caller)
                self.session.auth = (
                    self.options.get("basic_auth_user"),
                    self.options.get("basic_auth_password"),
                )
            self._login()

    def cred_dump(self):
        ret_val = {
            "user": self.options.get('user'),
            "password": self.options.get('password'),
            "allow_self_signed": self.options.get("disable_ssl_cert_check"),
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "access_token_datetime": self.access_token_datetime.timestamp(),
            "refresh_token_datetime": self.refresh_token_datetime.timestamp(),
            "api_version": self.api_version,
            "zm_version": self.zm_version,
        }
        return ret_val

    @staticmethod
    def _versiontuple(v):
        # https://stackoverflow.com/a/11887825/1361529
        return tuple(map(int, (v.split("."))))

    def get_session(self):
        return self.session

    def version(self):
        """Returns version of API and ZM

        Returns:
            dict: Version of API and ZM::

            {
                status: string # if 'error' then will also have 'reason'
                api_version: string # if status is 'ok'
                zm_version: string # if status is 'ok'
            }
        """
        if not self.authenticated:
            return {"status": "error", "reason": "not authenticated"}
        return {
            "status": "ok",
            "api_version": self.api_version,
            "zm_version": self.zm_version,
        }

    def tz(self):
        """Returns timezone of ZoneMinder server

        Returns:
           string: timezone of ZoneMinder server (or None if API not supported)
        """

        idx = min(len(stack()), 2)
        caller = getframeinfo(stack()[idx][0])
        if not self.zm_tz:
            url = f"{self.api_url}/host/gettimezone.json"

            try:
                r = self.make_request(url=url)
            except HTTPError as err:
                g.logger.error(
                    f"{lp} timezone API not found, relative timezones will be local time",
                    caller=caller,
                )
            else:
                self.zm_tz = r.get("tz")

        return self.zm_tz

    def authenticated(self):
        """True if login API worked

        Returns:
            boolean -- True if Login API worked
        """

        return self.authenticated

    # called in _make_request to avoid 401s if possible
    def _refresh_tokens_if_needed(self):

        # global GRACE
        if not (self.access_token_expires and self.refresh_token_expires):
            return
        tr = (self.access_token_datetime - datetime.datetime.now()).total_seconds()
        if tr >= GRACE:  # grace for refresh lifetime
            # g.logger.Debug(2, f"{lp} access token still has {tr/60:.2f} minutes remaining")
            return
        else:
            self._re_login()

    def _re_login(self):
        """Used for 401. I could use _login too but decided to do a simpler fn"""

        idx = min(len(stack()), 2)
        caller = getframeinfo(stack()[idx][0])
        # global GRACE
        if self._versiontuple(self.api_version) >= self._versiontuple("2.0"):
            # use tokens
            tr = (self.refresh_token_datetime - datetime.datetime.now()).total_seconds()
            if tr >= GRACE:  # 5 mins grace
                g.logger.debug(
                    2,
                    f"{lp} using refresh token to get a new auth, as refresh still has {tr / 60} minutes remaining",
                    caller=caller,
                )
                self.options["token"] = self.refresh_token
            else:
                g.logger.debug(
                    f"{lp} refresh token only has {tr}s of lifetime, need to re-login (user/pass)",
                    caller=caller,
                )
                self.options["token"] = None
        self._login()

    def _login(self):
        """This is called by the constructor. You are not expected to call this directly.

        Raises:
            err: reason for failure
        """

        idx = min(len(stack()), 2)
        caller = getframeinfo(stack()[idx][0])
        try:
            if self.api_url:
                url = f"{self.api_url}/host/login.json"
            else:
                raise ValueError("api_url not set!")
            if self.options.get("token"):
                if self.sanitize:
                    show_token = f"{self.options.get('token')[:15]}..."
                else:
                    show_token = self.options["token"]
                g.logger.debug(
                    f"{lp} found token, using for login [{show_token}]",
                    caller=caller,
                )
                data = {"token": self.options.get("token")}
                self.auth_enabled = True

            elif self.options.get("user") and self.options.get("password"):
                g.logger.debug(
                    f"{lp} no token found, trying user/pass for login", caller=caller
                )
                data = {
                    "user": self.options.get("user"),
                    "pass": self.options.get("password"),
                }
                self.auth_enabled = True

            else:
                g.logger.debug(
                    f"{lp} not using auth (no user/pass detected)", caller=caller
                )
                self.auth_enabled = False
                data = {}
                url = f"{self.api_url}/host/getVersion.json"
            # print(f"{lp}DBG: {data = }")

            r = self.session.post(url, data=data)
            if r.status_code == 401 and self.options.get("token") and self.auth_enabled:
                g.logger.debug(
                    f"{lp} using refresh token failed. Likely revoked, trying user/pass login",
                    caller=caller,
                )
                self.options["token"] = None
                data = {
                    "user": self.options.get("user"),
                    "pass": self.options.get("password"),
                }
                r = self.session.post(url, data=data)
            r.raise_for_status()

            rj = r.json()
            self.api_version = rj.get("apiversion")
            self.zm_version = rj.get("version")
            if self.auth_enabled:
                if self._versiontuple(self.api_version) >= self._versiontuple("2.0"):
                    g.logger.debug(
                        2,
                        f"{lp} detected API ver 2.0+, using token system",
                        caller=caller,
                    )
                    self.auth_type = 'token'
                    self.access_token = rj.get("access_token", "")
                    if rj.get("refresh_token"):
                        self.refresh_token = rj.get("refresh_token")
                    if rj.get("access_token_expires"):
                        self.access_token_expires = int(rj.get("access_token_expires"))
                        self.access_token_datetime = (
                                datetime.datetime.now()
                                + datetime.timedelta(seconds=self.access_token_expires)
                        )
                        g.logger.debug(
                            f"{lp} access token expires on: {self.access_token_datetime} ({self.access_token_expires}s)",
                            caller=caller,
                        )
                    if rj.get("refresh_token_expires"):
                        self.refresh_token_expires = int(
                            rj.get("refresh_token_expires")
                        )
                        self.refresh_token_datetime = (
                                datetime.datetime.now()
                                + datetime.timedelta(seconds=self.refresh_token_expires)
                        )
                        g.logger.debug(
                            f"{lp} refresh token expires on: {self.refresh_token_datetime} "
                            f"({self.refresh_token_expires}s)",
                            caller=caller,
                        )
                else:
                    g.logger.info(
                        f"{lp} using old credentials API. Recommended you upgrade to token API (ver 2.0+)",
                        caller=caller,
                    )
                    self.auth_type = 'basic'
                    self.legacy_credentials = rj.get("credentials")
                    if rj.get("append_password") == "1":
                        self.legacy_credentials = (
                                self.legacy_credentials + self.options.get("password")
                        )
            self.authenticated = True
            # print (vars(self.session))

        except HTTPError as err:
            g.logger.error(f"{lp} got API login error: {err}", caller=caller)
            self.authenticated = False
            raise err

    def get_apibase(self):
        return self.api_url

    def get_portalbase(self):
        return self.portal_url

    def get_creds(self):
        if not self.auth_enabled or not self.api_version:
            return ""
        if self._versiontuple(self.api_version) >= self._versiontuple("2.0"):
            return self.options.get('user'), self.options.get('password')
        else:
            # FIXME: need to get it to a tuple of ('basic user', 'basic pass') and append password mechanism
            return self.legacy_credentials

    def get_auth(self):
        if not self.auth_enabled or not self.api_version:
            return ""
        if self._versiontuple(self.api_version) >= self._versiontuple("2.0"):
            return f"token={self.access_token}"
        else:
            return self.legacy_credentials

    def get_all_event_data(self, event_id=None, update_frame_buffer_length=True):
        """Returns the data from an 'Event' API call.
If you do not supply it an event_id it will use the global event id.
        ZoneMinder returns 3 structures in the JSON response.
    - Monitor data - A dict containing data about the event monitor.
    - Event data - A dict containing all info about the current event.
    - Frame data - A list whose length is the current amount of frames in the frame buffer for the event.

    :param update_frame_buffer_length: (bool) If True, will update the frame_buffer_length (Default: True).
    :param event_id: (str/int) Optional, the event ID to query."""

        if not event_id:
            event_id = g.eid
        Event: Optional[Dict]
        Monitor: Optional[Dict]
        Frame: Optional[List]
        events_url = f"{self.get_apibase()}/events/{event_id}.json"
        try:
            g.api_event_response = self.make_request(url=events_url, quiet=True)
        except Exception as e:
            g.logger.error(f"{lp} Error during Event data retrieval: {str(e)}")
            raise e
        else:
            Event = g.api_event_response.get("event", {}).get("Event")
            Monitor = g.api_event_response.get("event", {}).get("Monitor")
            Frame = g.api_event_response.get("event", {}).get("Frame")
            g.config["mon_name"] = Monitor.get("Name")
            g.config["api_cause"] = Event.get("Cause")
            g.config["eventpath"] = Event.get("FileSystemPath")
            if update_frame_buffer_length:
                g.event_tot_frames = len(Frame)
            # g.logger.Debug(f"{Event}")
            # g.logger.Debug(f"{Monitor}")
            # g.logger.Debug(f'{Frame}')
            # g.logger.Debug(f"{type(g.api_event_response.get('event', {}).get('Frame'))=} -- {g.api_event_response.get('event', {}).get('Frame')=}")
            return Event, Monitor, Frame

    def make_request(
            self,
            url=None,
            query=None,
            payload=None,
            type_action="get",
            reauth=True,
            quiet=False,
    ) -> dict:
        """
        :rtype: dict
        :rtype: object
        """

        idx = min(len(stack()), 1)
        caller = getframeinfo(stack()[idx][0])
        if payload is None:
            payload = {}
        if query is None:
            query = {}
        self._refresh_tokens_if_needed()
        type_action = type_action.lower()
        if self.auth_enabled:
            if self._versiontuple(self.api_version) >= self._versiontuple("2.0"):
                query["token"] = self.access_token

            else:
                # credentials are already query formatted
                lurl = url.lower()
                if lurl.endswith("json") or lurl.endswith("/"):
                    qchar = "?"
                else:
                    qchar = "&"
                url += qchar + self.legacy_credentials

        try:
            from pyzm.helpers.pyzm_utils import str2bool

            portal = self.portal_url
            if self.api_url and not portal:
                portal = self.api_url("api_portal")[:-4]
            show_url = str(url).replace(portal, f"{g.config['sanitize_str']}") if self.sanitize else url
            show_tkn = query.get('token')[:25] if self.sanitize else query.get('token')
            g.logger.debug(
                2,
                f"{lp}make_req: '{type_action}'->{show_url}{' payload={}'.format(payload) if len(payload) > 0 else ''} "
                f"query={query if not query.get('token') else {'token': '{}...'.format(show_tkn)} }",
                caller=caller,
            ) if not quiet else None
            if type_action == "get":
                r = self.session.get(url, params=query)
            elif type_action == "post":
                r = self.session.post(url, data=payload, params=query)
            elif type_action == "put":
                r = self.session.put(url, data=payload, params=query)
            elif type_action == "delete":
                r = self.session.delete(url, data=payload, params=query)
            else:
                g.logger.error(
                    f"{lp}make_req: unsupported request type:{type_action}",
                    caller=caller,
                )
                raise ValueError(
                    f"{lp}make_req: unsupported request type:{type_action}"
                )
            r.raise_for_status()
            # Empty response, e.g. to DELETE requests, can't be parsed to json
            # even if the content-type says it is application/json

            if r.headers.get("content-type").startswith("application/json") and r.text:
                return r.json()
            elif r.headers.get("content-type").startswith("image/"):
                return r
            else:
                # A non 0 byte response will usually mean its an image eid request that needs re-login
                if r.headers.get("content-length") != "0":
                    g.logger.debug(4, f"{lp} raising RELOGIN ValueError", caller=caller)
                    raise ValueError("RELOGIN")
                else:
                    # ZM returns 0 byte body if index not found (no frame ID or out of bounds)
                    g.logger.debug(
                        4,
                        f"{lp} raising BAD_IMAGE ValueError as Content-Length:0 (OOB or bad frame ID)",
                        caller=caller,
                    )
                    raise ValueError("BAD_IMAGE")
                # return r.text

        except HTTPError as err:
            if err.response.status_code == 401 and reauth:
                g.logger.debug(
                    f"{lp} Got 401 (Unauthorized) - retrying auth login once",
                    caller=caller,
                )
                self._re_login()
                g.logger.debug(f"{lp} Retrying failed request again...", caller=caller)
                return self.make_request(url, query, payload, type_action, reauth=False)
            elif err.response.status_code == 404:
                err.response: requests.Response
                err_json: Optional[dict] = err.response.json()
                if err_json:
                    g.logger.error(f"{lp} JSON ERROR response >>> {err_json}")
                    if err_json.get('success') is False:
                        # get the reason instead of guessing
                        err_name = err_json.get('data').get('message')
                        err_message = err_json.get('data').get('message')
                        err_url = err_json.get('data').get('url')
                        if err_name == 'Invalid event':
                            g.logger.debug(f"{lp} raising Invalid Event", caller=caller)
                            raise ValueError("Invalid Event")
                        else:
                            # ZM returns 404 when an image cannot be decoded or the requested event does not exist
                            g.logger.debug(
                                4,
                                f"{lp} raising BAD_IMAGE ValueError for a 404 (image does not exist)",
                                caller=caller,
                            )
                            raise ValueError("BAD_IMAGE")
            else:
                err_msg = (
                    str(err).replace(self.portal_url, f"{g.config['sanitize_str']}")
                    if g.config.get("sanitize_logs")
                    else err
                )
                g.logger.debug(f"{lp} HTTP error: {err_msg}", caller=caller)
        except ValueError as err:
            err_msg = f"{err}"
            if err_msg == "RELOGIN":
                if reauth:
                    g.logger.debug(
                        f"{lp} got ValueError access error: {err}",
                        caller=caller,
                    )
                    g.logger.debug(f"{lp} retrying login once", caller=caller)
                    self._re_login()
                    g.logger.debug(
                        f"{lp} retrying failed request again...", caller=caller
                    )
                    return self.make_request(
                        url, query, payload, type_action, reauth=False
                    )
            else:
                raise err


    def zones(self, options=None):
        """Returns list of zones. Given zones are fairly static, maintains a cache and returns from cache on subsequent calls.

            Args:
                options (dict, optional): Available fields::

                    {
                        'force_reload': boolean # if True refreshes zones

                    }

        Returns:
            list of :class:`pyzm.helpers.Zone`: list of zones
        """
        
        if options is None:
            options = {}
        if options.get("force_reload") or not self.Zones:
            self.Zones = Zones(globs=g)
        return self.Zones

    def monitors(self, options=None):
        """Returns list of monitors. Given monitors are fairly static, maintains a cache and returns from cache on subsequent calls.

            Args:
                options (dict, optional): Available fields::

                    {
                        'force_reload': boolean # if True refreshes monitors

                    }

        Returns:
            list of :class:`pyzm.helpers.Monitor`: list of monitors
        """
        
        if options is None:
            options = {}
        if options.get("force_reload") or not self.Monitors:
            self.Monitors = Monitors(globs=g)
        return self.Monitors

    def events(self, options=None):
        """Returns list of events based on filter criteria. Note that each time you called events, a new HTTP call is made.

        Args:
            options (dict, optional): Various filters that will be applied to events. Defaults to {}. Available fields::

                {
                    'event_id': string # specific event ID to fetch
                    'tz': string # long form timezone (example America/New_York),
                    'from': string # string # minimum start time (including human readable
                                   # strings like '1 hour ago' or '10 minutes ago to 5 minutes ago' to create a range)
                    'to': string # string # maximum end time
                    'mid': int # monitor id
                    'min_alarmed_frames': int # minimum alarmed frames
                    'max_alarmed_frames': int # maximum alarmed frames
                    'object_only': boolean # if True will only pick events
                                           # that have objects

                }

        Returns:
            list of :class:`pyzm.helpers.Event`: list of events that match criteria
        """
        
        if options is None:
            options = {}
        self.Events = Events(options=options, globs=g)
        return self.Events

    def states(self, options=None):
        """Returns configured states

        Args:
            options (dict, optional): Not used. Defaults to {}.

        Returns:
            list of  :class:`pyzm.helpers.State`: list of states
        """
        
        if options is None:
            options = {}
        self.States = States(globs=g)
        return self.States

    def restart(self):
        """Restarts ZoneMinder

        Returns:
            json: json value of restart command
        """
        return self.set_state(state="restart")

    def stop(self):
        """Stops ZoneMinder

        Returns:
            json: json value of stop command
        """
        return self.set_state(state="stop")

    def start(self):
        """Starts ZoneMinder

        Returns:
            json: json value of start command
        """
        return self.set_state(state="start")

    def set_state(self, state):
        """Sets Zoneminder state to specific state

        Args:
            state (string): Name of state

        Returns:
            json: value of state change command
        """
        
        if not state:
            return
        url = f"{self.api_url}/states/change/{state}.json"
        return self.make_request(url=url)

    def configs(self, options=None):
        """Returns config values of ZM

            Args:
                options (dict, optional): Defaults to {}.
                options::

                    {
                        'force_reload': boolean # if True, reloads
                    }

        Returns:
            :class:`pyzm.helpers.Configs`: ZM configs
        """
        
        if options is None:
            options = {}
        if options.get("force_reload") or not self.Configs:
            self.Configs = Configs(globs=g)
        return self.Configs
