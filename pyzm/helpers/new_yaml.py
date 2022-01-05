"""The new experimental config file parser for Neo-ZMEventNotification using YAML syntax.
Offers superior nested python data structure retention compared to the old ConfigParser logic.
"""
import glob
import os
import sys
import time
from ast import literal_eval
from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path
from re import compile
from shutil import which
from threading import Thread
from traceback import format_exc
from typing import Optional


from yaml import SafeLoader, load
from pyzm.helpers.pyzm_utils import my_stderr, my_stdout, str2bool
from pyzm.interface import GlobalConfig
from pyzm.api import ZMApi
from pyzm.ZMLog import ZMLog

ZM_INSTALLED: Optional[str] = which('zmdc.pl')
lp: str = "config:"
g: GlobalConfig

SECRETS_REGEX = r"^\b|\s*(\w.*):\s*\"?|\'?({\[\s*(\w.*)\s*\]})\"?|\'?"
SUBVAR_REGEX = r"^\b|\s*(\w.*):\s*\"?|\'?({{\s*(\w.*)\s*}})\"?|\'?"

class ConfigParse:
    """A class to parse and store ZMES and MLAPI config and secret files.
    Once you process the config file and optionally the per monitor overrides, the object holds several copies
    of the config in different states. There is a hash function to get the hash of the config file or the secrets file
    which can be stored and there is another function to check the old hash to a current hash to see if the file has changed.

    - A copy of the original file as it was read (with {[secrets]} and {{vars}}).
    - A copy of the config after the secrets and sub vars were substituted.
    - If you create overridden configs based on the configured monitors in the config file, there is a copy for each monitor.


    """
    config_hash: Optional[str] = None
    secrets_hash: Optional[str] = None

    @staticmethod
    def compute_file_checksum(path: str, read_chunk_size: int = 65536, algorithm: str = 'sha256'):
        """Compute checksum of a file's contents.

        :param path: Path to the file
        :param read_chunk_size: Maximum number of bytes to be read from the file
         at once. Default is 65536 bytes or 64KB
        :param algorithm: The hash algorithm name to use. For example, 'md5',
         'sha256', 'sha512' and so on. Default is 'sha256'. Refer to
         hashlib.algorithms_available for available algorithms
        :return: Hex digest string of the checksum

        """
        from hashlib import new
        checksum = new(algorithm)  # Raises appropriate exceptions.
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(read_chunk_size), b''):
                checksum.update(chunk)
        return checksum.hexdigest()

    def hash(self, filetype: str):
        """hash the config or secrets file based on **filetype**.
        :param filetype: (str) one of config or secret
        """
        def _compute(name):
            if name and Path(name).exists() and Path(name).is_file():
                try:
                    hash_obj = self.compute_file_checksum(name)
                except Exception as exc:
                    print(f"{lp}hash: ERROR while SHA-256 hashing '{Path(name).name}' -> {exc}")
                else:
                    g.logger.debug(f"{lp}hash: the SHA-256 hex digest for file '{name}' -> {hash_obj}")
                    return hash_obj

        if filetype == 'config':
            self.config_hash = _compute(self.config_file_name)
        elif filetype == 'secret':
            self.secrets_hash = _compute(self.secret_file_name)
        else:
            print(
                f"{lp}hash: incorrect parameter 'filetype' -> '{filetype}', it can only be 'config' or 'secret'")

    def hash_compare(self, filetype: str) -> bool:
        """Compares a cached hash to a new hash of the supplied *filetype*"""
        def _compare(filename, cached_hash):
            if filename and Path(filename).exists() and Path(filename).is_file():
                current_file_hash = self.compute_file_checksum(filename)
                # g.logger.debug(
                #     f"{lp}{self.type}:hash: 'current hash of the file': {current_file_hash} --- "
                #     f"'cached hash': {cached_hash}"
                # )
                return cached_hash == current_file_hash

        if filetype == 'config':
            if self.config_hash:
                return _compare(self.config_file_name, self.config_hash)
            else:
                print(f"{lp}{self.type}:hash: there is no cached hash for a configuration file")
        elif filetype == 'secret':
            if self.secrets_hash:
                return _compare(self.secret_file_name, self.secrets_hash)
            else:
                print(f"{lp}{self.type}:hash: there is no cached hash for a secrets file")
        else:
            print(
                f"{lp}{self.type}:hash: incorrect parameter 'filetype' -> '{filetype}', allowed "
                f"values: 'config' or 'secret'"
            )
        return False

    def process_config(self, *args_, **kwargs):
        def _base_key_prep():
            # Replace the {{vars}} in the base keys
            # Example: tpu_object_weights_mobiledet =
            # {{base_data_path}}/models/coral_edgetpu/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite
            # If the base keys don't have the vars replaced then the un-subbed vars will follow along.
            g.logger.debug(f"{lp}{self.type}:proc: substituting '{{{{variables}}}}' for the 'base' config keys")
            base_vars_replaced = []
            base_vars_not_replaced = []
            skip_sections = {'stream_sequence', 'ml_sequence', 'monitors', 'ml_routes', 'zmes_keys'}
            for def_key, def_value in self.config.items():
                if def_key in skip_sections:
                    continue
                else:
                    new_val = str(def_value)
                    _vars = compile(r'{{(\w*)}}').findall(new_val)
                    if _vars:
                        for var in _vars:
                            if var and var in self.config:
                                base_vars_replaced.append(var)
                                new_val = compile(r'(\{{\{{{key}\}}\}})'.format(key=var)).sub(str(self.config[var]),
                                                                                              new_val)
                                self.config[def_key] = new_val
                            else:
                                base_vars_not_replaced.append(var)

            if base_vars_replaced:
                base_vars_replaced = list(set(base_vars_replaced))
                g.logger.debug(
                    f"{lp}{self.type}: successfully replaced {len(base_vars_replaced)} default sub var"
                    f"{'' if len(base_vars_replaced) == 1 else 's'} in the base config -> "
                    f"{base_vars_replaced}"
                )
            if base_vars_not_replaced:
                base_vars_not_replaced = set(base_vars_not_replaced)
                g.logger.debug(
                    f"{lp}{self.type}: there {'was' if len(base_vars_not_replaced) == 1 else 'were'} "
                    f"{len(base_vars_not_replaced)} secret{'' if len(base_vars_not_replaced) == 1 else 's'}"
                    f"  configured that have no substitution in the base config -> {base_vars_not_replaced}"
                )

        # -----------------------------
        #           MAIN
        # -----------------------------
        # This is the config without secrets and substitution variables replaced

        dc: dict = self.default_config
        self.config = deepcopy(self.default_config)
        # iterate the keys in the built-in default values: if the default config file does not have a key that is in
        # the defaults, copy over the default key: value.
        def_keys_added = []
        if not self.config:
            print(
                f"{lp}{self.type}:proc: There was an error reading from the configuration file "
                f"'{self.config_file_name}' please check the formatting of the file!"
            )
            # todo pushover notification of failure
            exit(1)
        for default_key, default_value in self.builtin_default_config.items():
            if default_key not in self.config or (self.config.get(default_key) and not self.config[default_key]):
                def_keys_added.append(default_key)
                self.config[default_key] = default_value
        if def_keys_added:
            g.logger.debug(
                f"{lp}{self.type}:proc: {len(def_keys_added)} built in default key"
                f"{'' if len(def_keys_added) == 1 else 's'} added to the 'base' config -> {def_keys_added}"
            )
        # The base keys need to have their {{vars}} replaced, this way secrets could have a {{var}} in its path
        _base_key_prep()
        # Substitute {[secrets]}
        self.secret_file_name = self.config.get('secrets')
        self.config = self.parse_secrets(config=self.config, filename=self.secret_file_name)
        # Building Base Config - {{VARS}} replacement
        self.config = self.parse_vars(config=self.config, filename=self.config_file_name, config_pool=self.config)
        # todo make configurable? so users can add complex data structures and have them safely evaluated
        eval_sections = {'pyzm_overrides', 'platerec_regions', 'poly_color', 'hass_people', }
        for e_sec in eval_sections:
            # convert specific keys into python structures
            if self.config.get(e_sec) and isinstance(self.config[e_sec], str):
                self.config[e_sec] = literal_eval(self.config[e_sec])
        g.logger.debug(
            f"{lp}{self.type}:proc: Base config has been built, all properly configured {{[secrets]}} "
            f"and {{{{vars}}}} have been replaced! Remember you must manually build per monitor overrode configurations"
        )

    def __init__(self, config_file_name: str, default: dict, type_: Optional[str] = None):
        # custom detection patterns per monitor
        self.detection_patterns: dict = {}
        # Polygon coords per monitor
        self.polygons = {}
        # Zones imported from ZM
        self.imported_zones = None
        # Zones defined in the config
        self.defined_zones = None
        # Defaults that will not change after being written
        self.default_monitor_overrides = None
        self.default_ml_sequence = None
        self.default_stream_sequence = None
        # The config before any base keys are replaced
        self.default_config = None
        self.builtin_default_config = default
        # secrets and config
        self.secret_file_name = None
        self.secrets = None
        self.config_file_name = None
        # The final base config after secret and var substitution
        self.config = None

        self.overrode = False
        # the config that has been overriden by the selected per monitor values
        self.override_config = None
        # A dict containing { monitor-id: { the whole self.config but overridden by monitor-id section } }
        self.monitor_overrides = {}
        # There is a copy of a full config for each monitor with secrets and vars replaced once you build it
        # You can instead only build 1 extra config that has been overrode if it is a ZMES run.
        # mlapi can hash the config file to see if anything has changed since the last detection and rebuild if so
        # same for secrets, if secrets hash has changed then rebuild the config.
        self.type = type_ if type_ else 'UNKNOWN'
        self.stream_seq = None
        self.ml_seq = None
        self.ml_routes = None
        self.zmes_keys = None
        self.config_file_name: str = config_file_name
        self.COCO = []
        # alias
        cfn: str = self.config_file_name
        # Figure out the type based on some of the default keys if it is not known
        if not self.type or self.type == 'UNKNOWN':
            bdc = self.builtin_default_config
            if bdc.get('mlapi_secret_key'):
                self.type = 'mlapi'
            elif bdc.get('create_animation') is not None:
                self.type = 'zmes'
        # Validate path and file
        if Path(cfn).is_file():
            g.logger.debug(f"{lp}init: the supplied config file exists -> '{config_file_name}'")
            if self.type == 'mlapi':
                # SHA-256 checksum the config file and cache the result
                self.hash('config')
            try:
                # Read the supplied config file into a python dict using a safe loader
                with open(config_file_name, 'r') as stream:
                    self.default_config = load(stream, Loader=SafeLoader)
            except TypeError as e:
                g.logger.error(f"{lp}init: the supplied config file is not valid YAML -> '{config_file_name}'")
                raise e
            except Exception as exc:
                # print as the logger is just a buffer until we initialize ZMLog
                print(f"{lp}init: error trying to load YAML in config file -> '{cfn}'")
                print(exc)
            else:
                dc: dict = self.default_config
                self.monitors = dc.get('monitors')
                self.default_ml_sequence = dc.get('ml_sequence')
                self.default_stream_sequence = dc.get('stream_sequence')
                self.default_monitor_overrides = dc.get('monitors')
                self.ml_routes = dc.get('ml_routes')
                self.zmes_keys = dc.get('zmes_keys')
                g.logger.debug(
                    f"{lp}:init: default configuration built (no secrets or substitution vars replaced, yet!)"
                )

        elif not Path(cfn).exists():
            print(f"{lp}init: the configuration file '{cfn}' does not exist!")
            return
        elif not Path(cfn).is_file():
            print(f"{lp}init: the configuration file '{cfn}' exists but it is not a file!")
            return

    def parse_secrets(self, config=None, filename=None):
        secrets_pool = None
        new_config = None
        secrets_replaced = []
        secrets_not_replaced = []
        # Validate the secrets file
        if filename:
            sfn = filename
            if Path(sfn).exists() and Path(sfn).is_file():
                g.logger.debug(
                    f"{lp}{self.type}: the configured secrets file exists and is a file -> '{sfn}'")
                if sfn:
                    if self.type == 'mlapi':
                        # SHA-256 checksum the config file and cache result
                        self.hash('secret')
                    g.logger.debug(f"{lp}{self.type}: starting '{{[secrets]}}' substitution")
                    try:
                        # Load secrets file into a python dict
                        with open(sfn, 'r') as stream:
                            secrets_pool = load(stream, Loader=SafeLoader)
                    except Exception as exc:
                        print(f"{lp} an exception occurred while trying to load YAML from '{sfn}'")
                        print(exc)
                        return
                    else:
                        # make the whole config a string so we can run regex substitutions
                        # this allows {[secrets]} and {{vars}} to be repplaced no matter where
                        # in the config file they are!
                        new_config = str(config)
                        secrets_regex = compile(r'\{\[(\w*)\]\}')
                        # remove duplicates by converting to a set that can be iterated / hashed
                        all_secs = set(secrets_regex.findall(new_config))
                        for sec in all_secs:
                            # g.logger.Debug(f"{type(sec)} --- {sec}")
                            if sec in secrets_pool and secrets_pool[sec]:
                                # the secret in the config file has a configured key: value in the secrets file
                                secrets_replaced.append(sec)
                                # create a pattern to replace the secret, has to be list for the index tracking
                                pattern = [r'\{\[', r'{}'.format(sec), r'\]\}']
                                pattern = r''.join(pattern)
                                # Replace the secret everywhere in the entire config
                                new_config = compile(pattern=pattern).sub(secrets_pool[sec], new_config)

                            else:
                                # The secret does not have a key: value configured in the secrets file
                                secrets_not_replaced.append(sec)
                        # Debug output
                        if secrets_replaced:
                            g.logger.debug(f"{lp}{self.type}: successfully replaced {len(secrets_replaced)} secret"
                                           f"{'' if len(secrets_replaced) == 1 else 's'} in the base config -> "
                                           f"{secrets_replaced}"
                                           )
                        if secrets_not_replaced:
                            g.logger.debug(
                                f"{lp}{self.type}: there {'is' if len(secrets_not_replaced) == 1 else 'are'} "
                                f"{len(secrets_not_replaced)} secret{'' if len(secrets_not_replaced) == 1 else 's'}"
                                f" configured that {'has' if len(secrets_not_replaced) == 1 else 'have'} no "
                                f"substitution candidate{'' if len(secrets_not_replaced) == 1 else 's'} in the "
                                f"base config or the secrets file -> {secrets_not_replaced}"
                            )

                        try:
                            # Convert the string back into a python dict
                            return literal_eval(new_config)
                        except ValueError:
                            print(
                                f"{lp}{self.type}:secrets: there is a formatting error in the config file, "
                                f"error converting to a python data structure! Please review your config, remember "
                                f"to always quote the '{{[secrets]}}', '{{{{variables}}}}' and strings that contain "
                                f"special characters '@&^%#$@(*@)(_#&*$%@#%'"
                            )
                            return
            elif not Path(sfn).exists():
                print(f"{lp}{self.type}: the configured secrets file does not exist -> '{sfn}'")
            elif not Path(sfn).is_file():
                print(
                    f"{lp}{self.type}: the configured secrets file exists but it is not a file! -> '{sfn}'")
        else:
            g.logger.debug(f"{lp}{self.type}: no secrets file configured")

    def parse_vars(self, config=None, filename=None, config_pool=None):
        if filename and Path(filename).exists() and Path(filename).is_file():
            g.logger.debug(f"{lp}{self.type}: starting '{{{{variable}}}}' substitution")
        new_config = str(config)
        # This pattern finds the secret and returns the key inside of the {[ ]}
        vars_regex = compile(r'{{(\w*)}}')
        all_vars = set(vars_regex.findall(new_config))
        vars_replaced = []
        vars_not_replaced = []
        for var in all_vars:
            if var in config:
                # The replacement variable has a key: value in the base config
                vars_replaced.append(var)
                # Compile and sub 1 liner to replace all occurrences of {{variable}} with the keys value
                new_config = compile(r'(\{{\{{{key}\}}\}})'.format(key=var)).sub(str(config[var]), new_config)
            else:
                # There is a {{variable}} but there is no configured key: value for it
                vars_not_replaced.append(var)
        if vars_replaced:
            g.logger.debug(f"{lp}{self.type}: successfully replaced {len(vars_replaced)} sub var"
                           f"{'' if len(vars_replaced) == 1 else 's'} in the base config -> {vars_replaced}"
                           )
            if vars_not_replaced:
                g.logger.debug(
                    f"{lp}{self.type}: there {'was' if len(vars_not_replaced) == 1 else 'were'} "
                    f"{len(vars_not_replaced)} secret{'' if len(vars_not_replaced) == 1 else 's'}"
                    f"  configured that {'has' if len(vars_not_replaced) == 1 else 'have'} no substitution in the "
                    f"base config -> {vars_not_replaced}"
                )
        try:
            return literal_eval(new_config)
        except ValueError:
            print(
                f"something is wrong with the config file formatting, make sure all of your {{[secrets]}} "
                f"and {{{{sub vars}}}} have quotes around them if they are by themselves or in a quoted string if "
                f"it is embedded into as a sub-string"
            )

    def monitor_override(self, mid):
        try:
            mid = int(mid)
        except TypeError:
            print(
                f"{lp}{self.type}:{mid}: the monitor id passed -> '{mid}' needs to be a whole"
                f" numerical value, cannot override config!"
            )
        else:
            illegal_keys = {
                'base_data_path',
                'mlapi_secret_key',
                'port',
                'processes',
                'db_path',
                'secrets',
                'config',
                'debug',
                'baredebug',
                'version',
                'bareversion'
            }
            g.logger.debug(
                f"{lp}{self.type}:{mid}: attempting to build an overrode config from monitor '{mid}' overrides")
            # first replace secrets if there are any but when replacing grab the values from the monitor
            # overrides first. If the monitors overrides don't have the key, grab it from the current config.
            # same thing for {{vars}} and then create the overrode config property
            dmo: dict = self.default_monitor_overrides
            if dmo and mid in dmo and dmo[mid] is not None:
                # Flag that there is an overrode config
                self.overrode = True
                # Convert to string to do the regex replacements
                new_overrides = str(dmo[mid])
                if self.monitor_overrides is not None:
                    self.override_config = self.monitor_overrides[mid] = deepcopy(self.config)
                secrets_replaced_by_overrides = []
                secrets_replaced_by_config = []
                secrets_not_replaced = []
                g.logger.debug(
                    f"{lp}{self.type}:{mid}: monitor '{mid}' overrides, starting '{{[secrets]}}' "
                    f"substitution"
                )
                secrets_regex = compile(r'\{\[(\w*)\]\}')
                all_secs = set(secrets_regex.findall(new_overrides))
                for sec in all_secs:
                    # g.logger.Debug(f"{type(sec)} --- {sec}")
                    if sec in dmo[mid]:
                        # First check if the secret has a key: value in the per monitor overrides
                        secrets_replaced_by_overrides.append(sec)
                        pattern = [r'\{\[', r'{}'.format(sec), r'\]\}']
                        pattern = r''.join(pattern)
                        new_overrides = compile(pattern=pattern).sub(dmo[mid][sec], new_overrides)
                    elif sec in self.config:
                        # If not check if the config has a key: value for the secret
                        secrets_replaced_by_config.append(sec)
                        pattern = [r'\{\[', r'{}'.format(sec), r'\]\}']
                        pattern = r''.join(pattern)
                        new_overrides = compile(pattern=pattern).sub(self.config[sec], new_overrides)
                    else:
                        secrets_not_replaced.append(sec)

                if secrets_replaced_by_overrides:
                    g.logger.debug(
                        f"{lp}{self.type}:{mid}: successfully replaced {len(secrets_replaced_by_overrides)} secret"
                        f"{'' if len(secrets_replaced_by_overrides) == 1 else 's'} from the OVERRIDES -> "
                        f"{secrets_replaced_by_overrides}"
                    )
                if secrets_replaced_by_config:
                    g.logger.debug(
                        f"{lp}{self.type}:{mid}: successfully replaced {len(secrets_replaced_by_config)} secret"
                        f"{'' if len(secrets_replaced_by_config) == 1 else 's'} from the OVERRIDES -> "
                        f"{secrets_replaced_by_config}"
                    )
                if secrets_not_replaced:
                    g.logger.debug(
                        f"{lp}{self.type}:{mid}: there {'was' if len(secrets_not_replaced) == 1 else 'were'} "
                        f"{len(secrets_not_replaced)} secret{'' if len(secrets_not_replaced) == 1 else 's'}"
                        f"  configured that have no substitution in the base config -> {secrets_not_replaced}"
                    )
                if not secrets_replaced_by_overrides and not secrets_replaced_by_config and not secrets_not_replaced:
                    g.logger.debug(f"{lp}{self.type}:{mid}: no secrets were replace during overrode config build")
                # {{VARS}}
                vars_regex = compile(r'{{(\w*)}}')
                all_vars = set(vars_regex.findall(new_overrides))
                vars_replaced_by_overrides = []
                vars_replaced_by_config = []
                vars_not_replaced = []
                for var in all_vars:
                    if var in dmo:
                        vars_replaced_by_overrides.append(var)
                        new_config = compile(r'(\{{\{{{key}\}}\}})'.format(key=var)).sub(str(dmo[var]), new_config)

                    if var in self.config:
                        vars_replaced_by_config.append(var)
                        new_config = compile(r'(\{{\{{{key}\}}\}})'.format(key=var)).sub(str(self.config[var]),
                                                                                         new_config)
                    else:
                        vars_not_replaced.append(var)
                if vars_replaced_by_overrides:
                    g.logger.debug(
                        f"{lp}{self.type}:{mid}: successfully replaced {len(vars_replaced_by_overrides)} sub var"
                        f"{'' if len(vars_replaced_by_overrides) == 1 else 's'} in the base config -> "
                        f"{vars_replaced_by_overrides}"
                    )
                if vars_replaced_by_config:
                    g.logger.debug(
                        f"{lp}{self.type}:{mid}: successfully replaced {len(vars_replaced_by_config)} sub var"
                        f"{'' if len(vars_replaced_by_config) == 1 else 's'} in the base config -> "
                        f"{vars_replaced_by_config}"
                    )
                if vars_not_replaced:
                    g.logger.debug(
                        f"{lp}{self.type}:{mid}: there {'was' if len(vars_not_replaced) == 1 else 'were'} "
                        f"{len(vars_not_replaced)} secret{'' if len(vars_not_replaced) == 1 else 's'}"
                        f"  configured that {'has' if len(vars_not_replaced) == 1 else 'have'} no substitution in the "
                        f"base config -> {vars_not_replaced}"
                    )
                if not vars_replaced_by_overrides and not vars_replaced_by_config and not vars_not_replaced:
                    g.logger.debug(f"{lp}{self.type}:{mid}: no vars were replace during overrode config build")
                try:
                    new_overrides = literal_eval(new_overrides)
                except ValueError:
                    print(
                        f"{lp}{self.type}:{mid}: there is a formatting error in the config "
                        f"file, error converting to a python data structure"
                    )
                else:
                    overrode = []
                    new_ = []
                    # convert coords to something Polygon can consume
                    from pyzm.helpers.pyzm_utils import str2tuple
                    # use Polygon to confirm proper coords
                    from shapely.geometry import Polygon
                    for overrode_key, overrode_val in new_overrides.items():
                        # Check for fuc*ery
                        if overrode_key in illegal_keys:
                            g.logger.debug(
                                f"{lp}{self.type}:{mid}: can not override '{overrode_key}' in monitor '{mid}' config, "
                                f"this may cause unexpected behavior and is off limits for per monitor overrides")
                            continue
                        # custom detection patterns for zones
                        elif overrode_key.endswith('_zone_detection_pattern'):
                            name = overrode_key.split('_zone_detection_pattern')[0]
                            self.detection_patterns[name] = overrode_val
                            if mid in self.polygons:
                                for idx, p in enumerate(self.polygons[mid]):
                                    if p['name'] == name:
                                        self.polygons[mid][idx]['pattern'] = overrode_val
                            g.logger.debug(f"{lp}{self.type}:{mid}: detection pattern for monitor '{mid}' defined "
                                           f"zone '{name}' -> {overrode_val}"
                                           )
                        # Custom defined Polygon / Zone
                        elif overrode_key.endswith('_polygon_zone') or overrode_key.endswith('_polygonzone'):
                            g.logger.debug(f"{lp}{self.type}:{mid}: polygon specified -> '{overrode_key}' for"
                                           f" monitor {mid}, validating polygon shape...")
                            try:
                                coords = str2tuple(overrode_val)
                                test = Polygon(coords)
                            except Exception as exc:
                                print(
                                    f"{lp}{self.type}:{mid}: the polygon coordinates supplied from '{overrode_key}' "
                                    f"are malformed! -> {overrode_val}"
                                )
                                exit()
                            else:
                                # works for _polygon_zone and _polygonzone
                                name = overrode_key.split('_polygon')[0]
                                pattern = self.detection_patterns[name] if name in self.detection_patterns else None
                                if mid in self.polygons:
                                    g.logger.debug(
                                        f"{lp}{self.type}:{mid}: appending to the existing entry in "
                                        f"polygons!"
                                    )
                                    self.polygons[mid].append(
                                        {
                                            'name': name,
                                            'value': coords,
                                            'pattern': pattern
                                        }
                                    )
                                else:
                                    g.logger.debug(f"{lp}{self.type}:{mid}: creating new entry in polygons!")
                                    self.polygons[mid] = [
                                        {
                                            'name': name,
                                            'value': coords,
                                            'pattern': pattern
                                        }
                                    ]
                                g.logger.debug(
                                    f"{lp}{self.type}:{mid}: successfully validated polygon for defined zone "
                                    f"'{name}' -> {coords}"
                                )
                        if overrode_key in self.override_config:
                            # There is a key in the config to override
                            overrode.append(overrode_key)
                        else:
                            # there is not a key to override so a new key will be created
                            new_.append(overrode_key)
                        self.override_config[overrode_key] = overrode_val

                    # Make a new deep copy of the overrode config for the monitor in the overrides dict
                    self.monitor_overrides[mid] = deepcopy(self.override_config)
                    if overrode:
                        g.logger.debug(
                            f"{lp}{self.type}:{mid}: {len(overrode)} keys overridden in the 'base' config "
                            f"-> {overrode}"
                        )
                    if new_:
                        g.logger.debug(
                            f"{lp}{self.type}:{mid}: {len(new_)} keys that did not have a 'base' value to override "
                            f"that are now in the 'base' config -> {new_}"
                        )


def start_logs(config: dict, args: dict, type_: str = 'unknown', no_signal: bool = False):
    # Setup logger and API, baredebug means DEBUG level logging but do not output to console
    # this is handy if you are monitoring the log files with tail -F (or the provided es.log.<detect/base> or mlapi.log)
    # otherwise you get double output. mlapi and ZMES override their std.out and std.err in order to catch all errors
    # and log them
    if args.get('debug') and args.get('baredebug'):
        g.logger.warning(f"{lp} both debug flags enabled! --debug takes precedence over --baredebug")
        args.pop('baredebug')

    if args.get('debug'):
        config['pyzm_overrides']['dump_console'] = True

    if args.get('debug') or args.get('baredebug'):
        config['pyzm_overrides']['log_debug'] = True
        if not config['pyzm_overrides'].get('log_level_syslog'):
            config['pyzm_overrides']['log_level_syslog'] = 5
        if not config['pyzm_overrides'].get('log_level_file'):
            config['pyzm_overrides']['log_level_file'] = 5
        if not config['pyzm_overrides'].get('log_level_debug'):
            config['pyzm_overrides']['log_level_debug'] = 5
        if not config['pyzm_overrides'].get('log_debug_file'):
            # log levels -> 1 dbg/print/blank 0 info, -1 warn, -2 err, -3 fatal, -4 panic, -5 off
            config['pyzm_overrides']['log_debug_file'] = 1

    if not ZM_INSTALLED:
        # Turn DB logging off if ZM is not installed
        config['pyzm_overrides']['log_level_db'] = -5

    if type_ == 'mlapi':
        log_path: str = ''
        log_name = 'zm_mlapi.log'
        if not ZM_INSTALLED:
            g.logger.debug(f"{lp}init:log: Zoneminder is not installed, configuring mlapi logger")
            if config.get('log_user'):
                log_user = config['log_user']
            if config.get('log_group'):
                log_group = config['log_group']
            elif not config.get('log_group') and config.get('log_user'):
                # use log user as log group as well
                log_group = config['log_user']
            log_path = f"{config['base_data_path']}/logs"
            # create the log dir in base_data_path, if it exists do not throw an exception
            Path(log_path).mkdir(exist_ok=True)

        elif ZM_INSTALLED:
            g.logger.debug(f"{lp}init:log: Zoneminder is installed, configuring mlapi logger")
            # get the system's apache user (www-data, apache, etc.....)
            from pyzm.helpers.pyzm_utils import get_www_user
            log_user, log_group = get_www_user()
            # fixme: what if system logs are elsewhere?
            if Path("/var/log/zm").is_dir():
                print(f"TESTING! mlapi is on same host as ZM, using '/var/log/zm' as logging path")
                log_path = "/var/log/zm"
            else:
                print(f"TESTING! mlapi is on the same host as ZM but '/var/log/zm' is not created or inaccessible, "
                      f"using the configured (possibly default) log path '{config['base_data_path']}/logs'")
                log_path = f"{config['base_data_path']}/logs"
                # create the log dir in base_data_path, if it exists do not throw an exception
                Path(log_path).mkdir(exist_ok=True)

        else:
            print(f"It seems there is no user to log with, there will only be console output, if anything"
                  f" at all. Configure log_user and log_group in your config file -> '{args.get('config')}'")
            log_user = None
            log_group = None

        log_name = config.get('log_name', log_name)
        # Validate log path if supplied in args
        if args.get('log_path'):
            if args.get('log_path_force'):
                g.logger.debug(f"{lp}init: 'force_log_path' is enabled!")
                Path(args.get('log_path')).mkdir(exist_ok=True)
            else:
                log_p = Path(args.get('log_path'))
                if log_p.is_dir():
                    log_path = args.get('log_path')
                elif log_p.exists() and not log_p.is_dir():
                    print(
                        f"{lp}init: the specified 'log_path' ({log_p.name}) exists BUT it is not a directory! using "
                        f"'{log_path}'.")
                elif not log_p.exists():
                    print(
                        f"{lp}init: the specified 'log_path' ({log_p.name}) does not exist! using '{log_path}'.")

        config['pyzm_overrides']['logpath'] = log_path
        config['pyzm_overrides']['webuser'] = log_user
        config['pyzm_overrides']['webgroup'] = log_group

    elif type_ == 'zmes':
        log_name = 'zmesdetect.log'
        if args.get('monitor_id'):
            log_name = f"zmesdetect_m{args.get('monitor_id')}"
        elif args.get('file'):
            log_name = "zmesdetect_file"
        elif g.mid:
            log_name = f"zmesdetect_m{g.mid}"
        elif args.get('from_face_train'):
            log_name = "zmes_train_face"
    else:
        log_name = 'zmes_external'
        if args.get('logname'):
            log_name = args.get('logname')
    # print(f"DBG>> before intializing ZMLog -> pyzm_overrides = {config['pyzm_overrides']}")
    if not isinstance(g.logger, ZMLog):
        g.logger = ZMLog(name=log_name, override=config['pyzm_overrides'], globs=g, no_signal=no_signal)
    # print(f"DBG>> AFTER {g.logger.get_config()}")


def process_config(
        args: dict,
        type_: str
):
    # Singleton dataclass should already be instantiated.
    global g
    g = GlobalConfig()
    if args.get('from_docker') or args.get('docker'):
        g.config['DOCKER'] = True
    g.config['sanitize_str'] = '<sanitized>'
    # build default config, pass filename
    defaults = g.DEFAULT_CONFIG
    config_obj = ConfigParse(args['config'], defaults)
    config_obj.process_config()
    # config_obj.COCO = pop_coco_names(config_obj.config['yolo4_object_labels']
    g.config = config_obj.config
    if type_ == 'mlapi':
        # Need to build defined per monitors config
        for mon, _ in config_obj.monitors.items():
            if _ is not None:
                config_obj.monitor_override(mon)
        # Ensure setting resize in mlapi config file will not have any effect
        # stream options will override this if resize is set in the zmes config
        if config_obj.config.get('resize') is not None:
            config_obj.config.pop('resize')

    return config_obj, g


def create_api(args: dict):
    lp = 'zmes:api create:'
    g.logger.debug(f"{lp} building ZM API Session")
    # get the api going
    api_options = {
        "apiurl": g.config.get("api_portal"),
        "portalurl": g.config.get("portal"),
        "user": g.config.get("user"),
        "password": g.config.get("password"),
        "basic_auth_user": g.config.get("basic_user"),
        "basic_auth_password": g.config.get("basic_password"),
        "logger": g.logger,  # currently just a buffer that needs to be iterated and displayed
        "disable_ssl_cert_check": str2bool(g.config.get("allow_self_signed")),
        "sanitize_portal": str2bool(g.config.get("sanitize_logs")),
    }
    try:
        g.api = ZMApi(options=api_options, api_globals=g)
    except Exception as e:
        g.logger.error(f"{lp} {e}")
        raise e
    else:
        # get and set the monitor id, name, eventpath
        if args.get("eventid"):
            # set event id globally first before calling api event data
            g.config['eid'] = g.eid = args['eventid']
            # api call for event data
            g.Event, g.Monitor, g.Frame = g.api.get_all_event_data()
            g.config["mon_name"] = g.Monitor.get("Name")
            g.config["api_cause"] = g.Event.get("Cause")
            g.logger.debug(f"")
            g.logger.debug(f'{g.config["mon_name"] = } ---- {g.Monitor.get("Name") = }')
            g.logger.debug(f'{g.config["api_cause"] = } ----- {g.Event.get("Cause") = }')
            g.logger.debug(f"")
            if not args.get('reason'):
                args['reason'] = g.Event.get("Notes")
            g.config['mid'] = g.mid = args["monitor_id"] = int(g.Monitor.get("Id"))
            if args.get("eventpath", "") == "":
                g.config["eventpath"] = args["eventpath"] = g.Event.get("FileSystemPath")
            else:
                g.config["eventpath"] = args["eventpath"] = args.get("eventpath")
        g.logger.debug(f"{lp} ZM API created")
