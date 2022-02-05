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
from inspect import getframeinfo, stack, Traceback
from traceback import format_exc
from typing import Optional, Dict, List, Union

import requests
from requests import Session, Response
from requests.exceptions import HTTPError
from requests.packages.urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

from pyzm.helpers.Configs import Configs
from pyzm.helpers.Events import Events
from pyzm.helpers.Monitors import Monitors
from pyzm.helpers.States import States
from pyzm.helpers.Zones import Zones
from pyzm.interface import GlobalConfig

g: GlobalConfig
GRACE: int = 60 * 5  # 5 mins
lp: str = "ZM API:"


class ZMApi:
    def __init__(self, options: Optional[dict] = None, kickstart: Optional[dict] = None):
        """
        options (dict):
            - apiurl (str) - the full API URL (example https://server/zm/api)
            - portalurl (str) - the full portal URL (example https://server/zm)
            - user (str) - username (None if using 'no auth' - API will try and figure it out)
            - password (str) - password (None if using 'no auth' - API will try and figure it out)
            - disable_ssl_cert_check (bool) - if True will let you use self-signed certs
            - basic_auth_user (str) - basic auth username
            - basic_auth_password (str) - basic auth password




        kickstart - (dict) containing existing JWT token data supplied to MLAPI by ZMES (saves time by skipping login).
              - user (str): Username,
              - password (str): Password,
              - access_token (str): Access token,
              - refresh_token (str): Refresh token,
              - auth_type (str): Auth type (token, basic, None),
              - access_token_expires (str|int): Access token expiration time in seconds (Example: 3600),
              - refresh_token_expires (str|int): Refresh token expiration time in seconds (Example: 3600),
              - access_token_datetime (str|float): Access token datetime in timestamp format,
              - refresh_token_datetime (str|float): Refresh token datetime in timestamp format,
              - api_version (str): API version,
              - zm_version (str): ZoneMinder version,
        """
        global g
        g = GlobalConfig()
        lp: str = "ZM API:init:"
        idx: int = min(len(stack()), 1)
        caller: Union[str, Traceback] = getframeinfo(stack()[idx][0])
        if options is None:
            g.logger.debug(f"{lp}:init no options were passed for initialization")
            options = {}
        self.api_url: Optional[str] = options.get("apiurl")
        self.portal_url: Optional[str] = options.get("portalurl")
        if not self.portal_url and (self.api_url and isinstance(self.api_url, str) and self.api_url.endswith("/api")):
            self.portal_url = self.api_url[: -len("/api")]
            g.logger.debug(
                2, f"{lp} portal not passed, guessing portal URL from portal_api is: {self.portal_url}", caller=caller
            )

        self.options: dict = options
        self.sanitize: bool = False
        self.auth_type: Optional[str] = None
        self.authenticated: bool = False
        self.auth_enabled: bool = False
        self.access_token: Optional[str] = ""
        self.refresh_token: Optional[str] = ""
        self.access_token_expires: Optional[str] = None
        self.refresh_token_expires: Optional[str] = None
        self.refresh_token_datetime: Optional[Union[datetime, str]] = None
        self.access_token_datetime: Optional[Union[datetime, str]] = None
        self.legacy_credentials: Optional[str] = None
        self.api_version: Optional[str] = ""
        self.zm_version: Optional[str] = ""
        self.zm_tz: Optional[str] = None

        self.Monitors: Optional[List[Monitors]] = None
        self.Events: Optional[List[Events]] = None
        self.Configs: Optional[List[Configs]] = None
        self.Zones: Optional[List[Zones]] = None
        self.States: Optional[List[States]] = None
        self.session: Session = Session()
        # Sanitize logs of urls, passwords, tokens, etc. Makes for easier copy+paste
        if self.options.get("sanitize_portal"):
            self.sanitize = True
        if self.options.get("disable_ssl_cert_check", True):
            self.session.verify = False
            g.logger.debug(
                2,
                f"{lp}init: SSL certificate verification disabled (encryption enabled, vulnerable to MITM attacks)",
                caller=caller,
            )
            disable_warnings(category=InsecureRequestWarning)

        if kickstart:
            self.options["user"] = kickstart.get("user")
            self.options["password"] = kickstart.get("password")
            self.auth_type = kickstart.get("auth_type")
            self.access_token = kickstart.get("access_token")
            self.refresh_token = kickstart.get("refresh_token")
            self.access_token_expires = kickstart.get("access_token_expires")
            self.refresh_token_expires = kickstart.get("refresh_token_expires")
            self.api_version = kickstart.get("api_version")
            self.zm_version = kickstart.get("zm_version")
            self.authenticated = True
            self.access_token_datetime = None
            self.refresh_token_datetime = None
            if kickstart.get("refresh_token_datetime"):
                self.refresh_token_datetime = datetime.datetime.fromtimestamp(
                    float(kickstart.get("refresh_token_datetime"))
                )
            if kickstart.get("access_token_datetime"):
                self.access_token_datetime = datetime.datetime.fromtimestamp(
                    float(kickstart.get("access_token_datetime"))
                )
                self.auth_enabled = True

                g.logger.debug(2, f"{lp}KICKSTART: a JWT and associated data has been supplied", caller=caller)
            else:
                g.logger.debug(
                    2,
                    f"{lp}KICKSTART: NO JWT has been supplied, assuming 'OPT_USE_AUTH' " f"is disabled",
                    caller=caller,
                )
        else:
            if self.options.get("basic_auth_user") and self.options.get("basic_auth_password"):
                g.logger.debug(4, f"{lp} BASIC auth requested, configuring...", caller=caller)
                self.session.auth = (
                    self.options.get("user"),
                    self.options.get("password"),
                )
            elif not self.options.get("basic_auth_user") and self.options.get("basic_auth_password"):
                g.logger.error(f"{lp} BASIC AUTH>>> password was supplied, user was not!")
            elif not self.options.get("basic_auth_password") and self.options.get("basic_auth_user"):
                g.logger.error(f"{lp} BASIC AUTH>>> user was supplied, password was not!")

            self._login()

    def cred_dump(self):
        # Use a template
        ret_val: dict = {
            "user": None,
            "password": None,
            "allow_self_signed": None,
            "access_token": None,
            "refresh_token": None,
            "access_token_datetime": None,
            "refresh_token_datetime": None,
            "api_version": None,
            "zm_version": None,
            "auth_type": self.auth_type,
            "enabled": self.auth_enabled,
        }
        if self.auth_enabled and self.auth_type == "token":
            try:
                ret_val["user"] = self.options.get("user")
                ret_val["password"] = self.options.get("password")
                ret_val["allow_self_signed"] = self.options.get("disable_ssl_cert_check")
                ret_val["access_token"] = self.access_token
                ret_val["refresh_token"] = self.refresh_token
                ret_val["access_token_datetime"] = self.access_token_datetime.timestamp()
                ret_val["refresh_token_datetime"] = self.refresh_token_datetime.timestamp()
                ret_val["api_version"] = self.api_version
                ret_val["zm_version"] = self.zm_version
            except Exception as e:
                g.logger.error(f"{lp} ERROR while attempting to dump current credentials")
                g.logger.debug(f"{lp} CRED DUMP DEBUG>>>  exception as str -> {e}")
                g.logger.debug(format_exc())
        elif not self.auth_enabled:
            g.logger.debug(
                f"{lp} Authentication is not enabled, no credentials will be passed to mlapi only portal data"
            )
            ret_val["allow_self_signed"] = self.options.get("disable_ssl_cert_check")
            ret_val["api_version"] = self.api_version
            ret_val["zm_version"] = self.zm_version
        elif self.auth_enabled and self.auth_type == "basic":
            g.logger.debug(f"BASIC AUTH CRED DUMP?!?!?!?!??!?")
        else:
            g.logger.error(f"API AUTH TYPE or AUTH ENABLED WEIRDNESS - {self.auth_enabled = } - {self.auth_type = }")
        return ret_val

    @staticmethod
    def _version_tuple(v):
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
        return {"status": "ok", "api_version": self.api_version, "zm_version": self.zm_version}

    def tz(self, force=False):
        """Returns timezone of ZoneMinder server

        :param force: (bool) - TZ is cached, use force=True to force an API query.
        Returns:
           string: timezone of ZoneMinder server (or None if API not supported)
        """

        idx = min(len(stack()), 2)
        caller = getframeinfo(stack()[idx][0])
        if not self.zm_tz or self.zm_tz and force:
            url = f"{self.api_url}/host/gettimezone.json"

            try:
                r = self.make_request(url=url)
            except HTTPError as err:
                g.logger.error(
                    f"{lp} timezone API not found, relative timezones will be local time",
                    caller=caller,
                )
                g.logger.debug(f"{lp} EXCEPTION>>> {err}")
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
        if self._version_tuple(self.api_version) >= self._version_tuple("2.0"):
            # use tokens
            time_remaining = (self.refresh_token_datetime - datetime.datetime.now()).total_seconds()
            if time_remaining >= GRACE:  # 5 mins grace
                g.logger.debug(
                    2,
                    f"{lp} using refresh token to get a new auth, as refresh still has {time_remaining / 60} "
                    f"minutes remaining",
                    caller=caller,
                )
                self.options["token"] = self.refresh_token
            else:
                g.logger.debug(
                    f"{lp} refresh token only has {time_remaining}s of lifetime, need to re-login (user/pass)",
                    caller=caller,
                )
                self.options["token"] = None
        self._login()

    def _login(self):
        """This is called by the constructor. You are not expected to call this directly.

        Raises:
            err: reason for failure
        """
        lp: str = "ZM API:login:"
        idx: int = min(len(stack()), 2)
        caller: Traceback = getframeinfo(stack()[idx][0])
        login_data: dict = {}
        if self.api_url:
            url = f"{self.api_url}/host/login.json"
        else:
            raise ValueError(f"{lp} api_url not set!")
        if self.options.get("token"):
            show_token = (
                f"{self.options['token'][:10]}...{g.config.get('sanitize_str')}"
                if self.sanitize
                else self.options["token"]
            )
            g.logger.debug(
                f"{lp} token was found, using for login -> [{show_token}]",
                caller=caller,
            )
            login_data = {"token": self.options["token"]}
        # token was not passed, check if user/password are supplied
        elif self.options.get("user") and self.options.get("password"):
            g.logger.debug(f"{lp} no token found, user/pass has been supplied, trying credentials...", caller=caller)
            login_data = {
                "user": self.options.get("user"),
                "pass": self.options.get("password"),
            }
        elif self.options.get("password") and not self.options.get("user"):
            g.logger.error(f"{lp} password was supplied but no user supplied, cannot login", caller=caller)
        elif self.options.get("user") and not self.options.get("password"):
            g.logger.error(f"{lp} user was supplied but no password supplied, cannot login", caller=caller)
        else:
            g.logger.debug(f"{lp} not using auth (no user/password was supplied)", caller=caller)
            url = f"{self.api_url}/host/getVersion.json"
        try:
            r = self.session.post(url, data=login_data)
            if r.status_code == 401 and self.options.get("token"):
                g.logger.debug(
                    f"{lp} token auth failed. Likely revoked, trying user/password login",
                    caller=caller,
                )
                self.options["token"] = None
                login_data = {
                    "user": self.options.get("user"),
                    "pass": self.options.get("password"),
                }
                r = self.session.post(url, data=login_data)
            r.raise_for_status()

            rj = r.json()
            self.api_version = rj.get("apiversion")
            self.zm_version = rj.get("version")
            if rj and rj.get("access_token"):
                # there is a JSON response and there is data in the access_token field
                g.logger.debug(
                    f"{lp} there is a JSON response from login attempt and an access_token " f"has been supplied"
                )
                self.auth_enabled = True
            elif rj and not rj.get("access_token"):
                if rj.get("credentials") and len(rj["credentials"]):
                    self.auth_enabled = True
                    self.auth_type = "legacy"
                    g.logger.warning(f"{lp} the API did not return a JWT but there are legacy credentials?")
                    self.legacy_credentials = rj["credentials"]
                    if rj.get("append_password") == "1":
                        g.logger.debug(
                            f"{lp} legacy credentials were returned and append_password is active, "
                            f"appending password to legacy credentials"
                        )
                        self.legacy_credentials = f"{self.legacy_credentials}&{self.options.get('password')}"
            elif not rj:
                g.logger.error(
                    f"{lp} there is not a response in JSON format after attempting a login" f", raising an error"
                )
                raise ValueError(f"{lp} No JSON response from login")

            if self.auth_enabled:
                if self._version_tuple(self.api_version) >= self._version_tuple("2.0"):
                    g.logger.debug(
                        2,
                        f"{lp} detected API ver 2.0+, using token system",
                        caller=caller,
                    )
                    self.auth_type = "token"
                    self.access_token = rj.get("access_token", "")
                    self.refresh_token = rj.get("refresh_token")
                    if rj.get("access_token_expires"):
                        self.access_token_expires = int(rj.get("access_token_expires"))
                        self.access_token_datetime = datetime.datetime.now() + datetime.timedelta(
                            seconds=self.access_token_expires
                        )
                        g.logger.debug(
                            f"{lp} access token expires on: {self.access_token_datetime} "
                            f"({self.access_token_expires}s)",
                            caller=caller,
                        )
                    if rj.get("refresh_token_expires"):
                        self.refresh_token_expires = int(rj.get("refresh_token_expires"))
                        self.refresh_token_datetime = datetime.datetime.now() + datetime.timedelta(
                            seconds=self.refresh_token_expires
                        )
                        g.logger.debug(
                            f"{lp} refresh token expires on: {self.refresh_token_datetime} "
                            f"({self.refresh_token_expires}s)",
                            caller=caller,
                        )
                else:
                    g.logger.warning(
                        f"{lp} using LEGACY API. Recommended you upgrade to token API (ver 2.0+)",
                        caller=caller,
                    )
                    g.logger.debug(f"{lp} API version is below '2.0' -> RESPONSE IN JSON -> {rj}")
                    self.auth_type = "legacy"
                    self.legacy_credentials = rj.get("credentials")
                    if rj.get("append_password") == "1":
                        self.legacy_credentials = f"{self.legacy_credentials}&{self.options.get('password')}"
            else:
                g.logger.debug(f"{lp} it is assumed 'OPT_USE_AUTH' is disabled!")
            self.authenticated = True

        except HTTPError as err:
            g.logger.error(f"{lp} got API login error: {err}", caller=caller)
            self.authenticated = False
            raise err

    def get_apibase(self):
        return self.api_url

    def get_portalbase(self):
        return self.portal_url

    def get_auth(self):
        if not self.auth_enabled or not self.api_version:
            return ""
        if self._version_tuple(self.api_version) >= self._version_tuple("2.0"):
            return f"token={self.access_token}"
        else:
            return self.legacy_credentials

    def get_all_event_data(self, event_id: int = None, update_frame_buffer_length: bool = True):
        """Returns the data from an 'Event' API call. If you do not supply it an event_id it will use the global event id.
            ZoneMinder returns 3 structures in the JSON response.
        - Monitor data - A dict containing data about the event' monitor.
        - Event data - A dict containing all info about the current event.
        - Frame data - A list whose length is the current amount of frames in the frame buffer for the event, also contains data about the frames.

        :param update_frame_buffer_length: (bool) If True, will update the frame_buffer_length (Default: True).
        :param event_id: (str/int) Optional, the event ID to query."""

        if not event_id:
            event_id = g.eid
        event: Optional[Dict] = None
        monitor: Optional[Dict] = None
        frame: Optional[List] = None
        events_url = f"{self.api_url}/events/{event_id}.json"
        try:
            g.api_event_response = self.make_request(url=events_url, quiet=True)
        except Exception as e:
            g.logger.error(f"{lp} Error during Event data retrieval: {str(e)}")
            g.logger.debug(f"{lp} EXCEPTION>>> {e}")
        else:
            event = g.api_event_response.get("event", {}).get("Event")
            monitor = g.api_event_response.get("event", {}).get("Monitor")
            frame = g.api_event_response.get("event", {}).get("Frame")
            g.config["mon_name"] = monitor.get("Name")
            g.config["api_cause"] = event.get("Cause")
            g.config["eventpath"] = event.get("FileSystemPath")
            if frame and update_frame_buffer_length:
                g.event_tot_frames = len(frame)
            return event, monitor, frame

    def make_request(
        self,
        url: Optional[str] = None,
        query: Optional[Dict] = None,
        payload: Optional[Dict] = None,
        type_action: str = "get",
        re_auth: bool = True,
        quiet: bool = False,
    ) -> Union[Dict, Response]:
        """
        :rtype: dict|Response
        """
        lp: str = "ZM API:make_req:"
        idx: int = min(len(stack()), 1)
        caller: Traceback = getframeinfo(stack()[idx][0])
        if payload is None:
            payload = {}
        if query is None:
            query = {}
        self._refresh_tokens_if_needed()
        type_action = type_action.lower()
        if self.auth_enabled:
            if self._version_tuple(self.api_version) >= self._version_tuple("2.0"):
                query["token"] = self.access_token
            else:
                # credentials are already query formatted
                lurl = url.lower()
                if lurl.endswith("json") or lurl.endswith("/"):
                    qchar = "?"
                else:
                    qchar = "&"
                url = f"{url}{qchar}{self.legacy_credentials}"
        try:
            if self.api_url and not self.portal_url:
                self.portal_url = self.api_url[:-4]
            show_url: str = url.replace(self.portal_url, g.config["sanitize_str"]) if self.sanitize else url
            show_tkn: str = (
                f"{query.get('token')[:10]}...{g.config.get('sanitize_str')}" if self.sanitize else query.get("token")
            )
            show_payload: str = ""
            show_query: str = f"token: '{show_tkn}'"
            if not query.get("token"):
                show_query = query
            if payload and len(payload):
                show_payload = f" payload={payload}"
            g.logger.debug(
                2,
                f"{lp} '{type_action}'->{show_url}{show_payload} query={show_query}",
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
                g.logger.error(f"{lp} unsupported request type: {type_action}", caller=caller)
                raise ValueError(f"Unsupported request type: {type_action}")
            r.raise_for_status()
            # Empty response, e.g. to DELETE requests, can't be parsed to json
            # even if the content-type says it is application/json

            if r.headers.get("content-type").startswith("application/json") and r.text:
                return r.json()
            elif r.headers.get("content-type").startswith("image/"):
                return r
            else:
                r: requests.Response
                # A non 0 byte response will usually mean it's an image eid request that needs re-login
                if r.headers.get("content-length") != "0":
                    g.logger.debug(4, f"{lp} raising RELOGIN ValueError", caller=caller)
                    g.logger.debug(f"{lp} DEBUG>>> {r.text = }")
                    raise ValueError("RELOGIN")
                else:
                    # ZM returns 0 byte body if index not found (no frame ID or out of bounds)
                    g.logger.debug(
                        4,
                        f"{lp} raising BAD_IMAGE ValueError as Content-Length: 0 (OOB or bad frame ID)",
                        caller=caller,
                    )
                    raise ValueError("BAD_IMAGE")
                # return r.text

        except HTTPError as err:
            # err.response: requests.Response
            if err.response.status_code == 401 and re_auth:
                g.logger.debug(
                    f"{lp} Got 401 (Unauthorized) - retrying auth login once -> {err.response.json()}",
                    caller=caller,
                )
                self._re_login()
                g.logger.debug(f"{lp} Retrying failed request again...", caller=caller)
                # recursion with a blocker
                return self.make_request(url, query, payload, type_action, re_auth=False)
            elif err.response.status_code == 404:
                err_json: Optional[dict] = err.response.json()
                if err_json:
                    g.logger.error(f"{lp} JSON ERROR response >>> {err_json}")
                    if err_json.get("success") is False:
                        # get the reason instead of guessing
                        err_name = err_json.get("data").get("message")
                        err_message = err_json.get("data").get("message")
                        err_url = err_json.get("data").get("url")
                        if err_name == "Invalid event":
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
            err_msg = f"{lp} EXCEPTION>>> {err}"
            if err_msg == "RELOGIN":
                if re_auth:
                    g.logger.debug(
                        f"{lp} got ValueError access error: {err}",
                        caller=caller,
                    )
                    g.logger.debug(f"{lp} retrying login once", caller=caller)
                    self._re_login()
                    g.logger.debug(f"{lp} retrying failed request again...", caller=caller)
                    return self.make_request(url, query, payload, type_action, re_auth=False)
            else:
                raise err

    def zones(self, options: Optional[dict] = None):
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
            self.Zones = Zones()
        return self.Zones

    def monitors(self, options: Optional[dict] = None):
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
            self.Monitors = Monitors()
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
        self.Events = Events(options=options)
        return self.Events

    def states(self):
        """Returns configured states

        Args:

        Returns:
            list of  :class:`pyzm.helpers.State`: list of states
        """
        self.States = States()
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

    def set_state(self, state: str):
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
            self.Configs = Configs()
        return self.Configs
