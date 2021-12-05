"""
ZMLog
=======
A python implementation of ZoneMinder's logging system
You can use this to connect it to the APIs if you want
"""
import copy
import glob
import grp
import os
from traceback import format_exc
from typing import Optional

import psutil
import pwd
import signal
import sys
import syslog
import time
from configparser import ConfigParser
from datetime import datetime
from dotenv import load_dotenv
from getpass import getuser
from inspect import getframeinfo, stack
from pathlib import Path
from shutil import which
from sqlalchemy import create_engine, MetaData, select, or_  # ,Table,inspect
from sqlalchemy.exc import SQLAlchemyError

g: Optional[object] = None
lp = 'ZM Log:'
zm_inst = which('zmdc.pl')


def sig_log_rot(sig, frame):
    lp = 'signal handler:'
    name = g.logger.logger_name
    overrides = g.logger.logger_overrides
    # close handlers
    g.logger.log_close()
    # re-init logger
    g.logger = ZMLog(name=name, override=overrides, no_signal=True)
    g.logger.info(f"{lp} ready after log re-initialization due to receiving HUP signal: {sig=}")


def sig_intr(sig, frame):
    lp = 'signal handler:'
    if sig == 2:
        g.logger.error(f"{lp} KeyBoard Interrupt -> 2:SIGINT received!")
    elif sig == 6:
        g.logger.error(f"{lp} Abort Interrupt -> 6:SIGABRT")
        g.logger.log_close()
        os.abort()
    else:
        g.logger.info(
            f"{lp} received interrupt signal ({sig}), safely closing logging handlers and exiting")
    if g.logger:
        # close the handlers, but don't remove them
        g.logger.log_close()
    exit(0)


levels = {
    'PRT': 1,
    'BLK': 1,
    'DBG': 1,
    'INF': 0,
    'WAR': -1,
    'ERR': -2,
    'FAT': -3,
    'PNC': -4,
    'OFF': -5
}

priorities = {
    'DBG': syslog.LOG_DEBUG,
    'INF': syslog.LOG_INFO,
    'WAR': syslog.LOG_WARNING,
    'ERR': syslog.LOG_ERR,
    'FAT': syslog.LOG_ERR,
    'PNC': syslog.LOG_ERR
}


class ZMLog:
    def __init__(self, name, override=None, caller=None, globs=None, no_signal=False):
        self.no_signal = no_signal
        global g
        g = globs
        self.buffer = g.logger
        self.is__active = None
        self.config = None
        self.engine = None
        self.conn = None
        self.sql_connected = False
        self.config_table = None
        self.log_table = None
        self.meta = None
        # File handler
        self.log_filename = None
        self.log_file_handler = None
        self.cstr = None
        load_dotenv()
        self.logger_name = None
        self.logger_overrides = None
        self.pid = None
        self.process_name = None
        self.syslog = None
        self.tot_errmsg = []
        self.exit_times_called: int = 0
        self.is_active = None
        self.initialize(name=name, override=override, caller=caller)


    def initialize(self, name, override=None, caller=None):
        if not override:
            override = {}
        self.config = {}
        self.syslog = syslog
        self.syslog.openlog(logoption=syslog.LOG_PID)
        self.pid = os.getpid()
        self.process_name = name or psutil.Process(self.pid).name()
        self.logger_name = name
        self.logger_overrides = override

        defaults = {
            'dbuser': None,
            'webuser': 'www-data',
            'webgroup': 'www-data',
            'logpath': '/var/log/zm',
            'log_level_syslog': 0,
            'log_level_file': 0,
            'log_level_db': 0,
            'log_debug': 0,
            'log_level_debug': 0,
            'log_debug_target': '',
            'log_debug_file': 0,
            'server_id': 0,

            'dump_console': False
        }

        self.config = {
            'conf_path': os.getenv('PYZM_CONFPATH', '/etc/zm'),  # we need this to get started
            'dbuser': os.getenv('PYZM_DBUSER'),
            'dbpassword': os.getenv('PYZM_DBPASSWORD'),
            'dbhost': os.getenv('PYZM_DBHOST'),
            'webuser': os.getenv('PYZM_WEBUSER'),
            'webgroup': os.getenv('PYZM_WEBGROUP'),
            'dbname': os.getenv('PYZM_DBNAME'),
            'logpath': os.getenv('PYZM_LOGPATH'),
            'log_level_syslog': os.getenv('PYZM_SYSLOGLEVEL'),
            'log_level_file': os.getenv('PYZM_FILELOGLEVEL'),
            'log_level_db': os.getenv('PYZM_DBLOGLEVEL'),
            'log_debug': os.getenv('PYZM_LOGDEBUG'),
            'log_level_debug': os.getenv('PYZM_LOGDEBUGLEVEL'),
            'log_debug_target': os.getenv('PYZM_LOGDEBUGTARGET'),
            'log_debug_file': os.getenv('PYZM_LOGDEBUGFILE'),
            'server_id': os.getenv('PYZM_SERVERID'),
            'dump_console': os.getenv('PYZM_DUMPCONSOLE'),
            'driver': os.getenv('PYZM_DBDRIVER', 'mysql+mysqlconnector')
        }
        # Round 1 of overrides, before we read params from DB
        # Override with locals if present
        self.config.update(override)
        if zm_inst:  # ZoneMinder is installed so use DB and read ZM' conf files
            if self.config.get('conf_path'):
                # read all config files in order
                files = []
                # Pythonic?
                # map(files.append, glob.glob(f'{self.config["conf_path"]}/conf.d/*.conf'))

                for f in glob.glob(f'{self.config["conf_path"]}/conf.d/*.conf'):
                    files.append(f)
                files.sort()
                files.insert(0, f"{self.config['conf_path']}/zm.conf")
                config_file = ConfigParser(interpolation=None, inline_comment_prefixes='#')
                f = None
                try:
                    for f in files:
                        with open(f, 'r') as s:
                            # print(f'reading {f}')
                            # This adds [zm_root] section to the head of each zm .conf.d config file,
                            # not physically only in memory
                            config_file.read_string(f'[zm_root]\n{s.read()}')
                except Exception as exc:
                    self.buffer.error(f"Error opening {f if f else files} -> {exc}")
                    self.buffer.error(f"{format_exc()}")
                    print(f"Error opening {f if f else files} -> {exc}")
                    print(f"{format_exc()}")
                    self.buffer.log_close(exit=1)
                    exit(1)
                else:
                    conf_data = config_file['zm_root']

                    if not self.config.get('dbuser'):
                        self.config['dbuser'] = conf_data.get('ZM_DB_USER')
                    if not self.config.get('dbpassword'):
                        self.config['dbpassword'] = conf_data.get('ZM_DB_PASS')
                    if not self.config.get('webuser'):
                        self.config['webuser'] = conf_data.get('ZM_WEB_USER')
                    if not self.config.get('webgroup'):
                        self.config['webgroup'] = conf_data.get('ZM_WEB_GROUP')
                    if not self.config.get('dbhost'):
                        self.config['dbhost'] = conf_data.get('ZM_DB_HOST')
                    if not self.config.get('dbname'):
                        self.config['dbname'] = conf_data.get('ZM_DB_NAME')
                    if not self.config.get('logpath'):
                        self.config['logpath'] = config_file['zm_root'].get('ZM_PATH_LOGS')
                    self.cstr = f"{self.config['driver']}://{self.config['dbuser']}:{self.config['dbpassword']}@" \
                                f"{self.config['dbhost']}/{self.config['dbname']}"

                    try:
                        self.engine = create_engine(self.cstr, pool_recycle=3600)
                        self.conn = self.engine.connect()
                        self.sql_connected = True
                    except SQLAlchemyError as e:
                        self.sql_connected = False
                        self.conn = None
                        self.engine = None
                        self.buffer.error(f"{lp} Turning DB logging off. Could not connect to DB, message was: {e}")
                        self.syslog.syslog(syslog.LOG_ERR, self._format_string(
                            f"Turning DB logging off. Could not connect to DB, message was: {e}"))
                        self.config['log_level_db'] = levels['OFF']

                    else:
                        self.meta = MetaData(self.engine, reflect=True)
                        self.config_table = self.meta.tables['Config']
                        self.log_table = self.meta.tables['Logs']

                        select_st = select([self.config_table.c.Name, self.config_table.c.Value]).where(
                            or_(self.config_table.c.Name == 'ZM_LOG_LEVEL_SYSLOG',
                                self.config_table.c.Name == 'ZM_LOG_LEVEL_FILE',
                                self.config_table.c.Name == 'ZM_LOG_LEVEL_DATABASE',
                                self.config_table.c.Name == 'ZM_LOG_DEBUG',
                                self.config_table.c.Name == 'ZM_LOG_DEBUG_LEVEL',
                                self.config_table.c.Name == 'ZM_LOG_DEBUG_FILE',
                                self.config_table.c.Name == 'ZM_LOG_DEBUG_TARGET',
                                self.config_table.c.Name == 'ZM_SERVER_ID',
                                ))
                        resultproxy = self.conn.execute(select_st)
                        db_vals = {row[0]: row[1] for row in resultproxy}
                        self.config['log_level_syslog'] = int(db_vals['ZM_LOG_LEVEL_SYSLOG'])
                        self.config['log_level_file'] = int(db_vals['ZM_LOG_LEVEL_FILE'])
                        self.config['log_level_db'] = int(db_vals['ZM_LOG_LEVEL_DATABASE'])
                        self.config['log_debug'] = int(db_vals['ZM_LOG_DEBUG'])
                        self.config['log_level_debug'] = int(db_vals['ZM_LOG_DEBUG_LEVEL'])
                        self.config['log_debug_file'] = db_vals['ZM_LOG_DEBUG_FILE']
                        self.config['log_debug_target'] = db_vals['ZM_LOG_DEBUG_TARGET']
                        self.config['server_id'] = db_vals.get('ZM_SERVER_ID', 0)
        else:  # There is no ZM installed on this system determined by zmdc.pl not being accessible in $PATH environment
            self.no_signal = True
            if self.buffer:
                self.buffer.info(f"(buffer)->ZoneMinder installation not detected, configuring mlapi accordingly...")
            else:
                self.info(f"ZoneMinder installation not detected, configuring mlapi accordingly...")
        # Round 2 of overrides, after DB data is read
        # Override with locals if present
        for key in defaults:
            if not self.config.get(key):
                self.config[key] = defaults[key]
        for key in self.config:
            if key in override:
                self.config[key] = override[key]
        # print ('FINAL CONFIGS {}'.format(self.config))
        # file handler
        self.log_filename = None
        self.log_file_handler = None
        # print('**********************A-R2 {}'. format(self.config))
        uid = pwd.getpwnam(self.config['webuser']).pw_uid
        gid = grp.getgrnam(self.config['webgroup']).gr_gid
        if self.config['log_level_file'] > levels['OFF']:
            self.log_filename = f"{self.config['logpath']}/{self.process_name}"
            try:
                log_file = Path(self.log_filename)
                log_file.touch(exist_ok=True)
                # Don't forget to close the file handler
                self.log_file_handler = log_file.open('a')
                os.chown(self.log_filename, uid, gid)  # proper permissions
            except Exception as e:
                self.buffer.error(f"{lp} Error for user: '{getuser()}' creating and changing permissions of file: "
                                  f"'{self.log_filename}'")
                self.buffer.error(f"{e}")
                self.syslog.syslog(
                    syslog.LOG_ERR,
                    self._format_string(
                        f"Error for user: {getuser()} while creating and changing permissions of log file -> {e}"
                    )
                )
        # Debug(f"File logging handler setup correctly -> {log_fhandle.name}") if log_fhandle else None
        if not self.no_signal:
            try:
                if self.buffer:
                    self.buffer.info(f"{lp}(buffer)->'Setting up signal handlers for log rotation and log interrupt'")
                else:
                    self.info(f"{lp}'Setting up signal handlers for log rotation and log interrupt'")
                signal.signal(signal.SIGHUP, sig_log_rot)
                signal.signal(signal.SIGINT, sig_intr)
            except Exception as e:
                if self.buffer:
                    self.buffer.error(f'{lp}(buffer)->Error setting up log rotate and interrupt signal handlers -> \n{e}\n')
                else:
                    self.error(f'Error setting up log rotate and interrupt signal handlers -> \n{e}\n')
        # This does some funky stuff when you thread log creation, instead just assign this object to g.logger
        # This is left over from when ZMLog was not a class but a bunch of functions
        # g.logger = sys.modules[__name__]

        self.is__active = True
        show_log_name = f"'{self.log_filename}'"
        if zm_inst:
            if self.buffer:
                self.buffer.info(
                    f"Connected to ZoneMinder Logging system with user '{getuser()}'"
                    f"{f' -> {show_log_name}' if self.log_file_handler else ''}"
                )
            else:
                self.info(
                    f"Connected to ZoneMinder Logging system with user '{getuser()}'"
                    f"{f' -> {show_log_name}' if self.log_file_handler else ''}"
                )
        else:
            if self.buffer:
                self.buffer.info(
                    f"Logging to {'syslog ' if self.syslog else ''}{'and ' if self.syslog and self.log_file_handler else ''}"
                    f"{'file ' if self.log_file_handler else ''}with user '{getuser()}'"
                    f"{f' -> {show_log_name}' if self.log_file_handler else ''}"
                )
            else:
                self.info(
                    f"Logging to {'syslog ' if self.syslog else ''}{'and ' if self.syslog and self.log_file_handler else ''}"
                    f"{'file ' if self.log_file_handler else ''}with user '{getuser()}'"
                    f"{f' -> {show_log_name}' if self.log_file_handler else ''}"
                )
        if self.buffer:
            for line in self.buffer:
                self.debug(line)

    def is_active(self):
        if self.is__active:
            return self.is__active

    def set_level(self, level):
        pass

    def get_buff(self):
        return

    def get_config(self, ):
        pass

    def _db_reconnect(self, ):
        try:
            self.conn.close()
        except Exception:
            pass
        if zm_inst:
            try:
                self.engine = create_engine(self.cstr, pool_recycle=3600)
                self.conn = self.engine.connect()
                # inspector = inspect(engine)
                # print(inspector.get_columns('Config'))
                self.meta = MetaData(self.engine, reflect=True)
                self.config_table = self.meta.tables['Config']
                self.log_table = self.meta.tables['Logs']
                message = 'reconnecting to Database...'
                log_string = '{level} [{pname}] [{msg}]'.format(level='INF', pname=self.process_name, msg=message)
                self.syslog.syslog(syslog.LOG_INFO, log_string)
            except SQLAlchemyError as e:
                self.sql_connected = False
                self.syslog.syslog(syslog.LOG_ERR,
                                   self._format_string("Turning off DB logging due to error received:" + str(e)))
                return False
            else:
                self.sql_connected = True
                return True

    def log_close(self, *args, **kwargs):
        idx = min(len(stack()), 1)  # in case someone calls this directly
        caller = getframeinfo(stack()[idx][0])

        self.exit_times_called += 1
        self.is__active = False
        if self.exit_times_called <= 1:
            if self.conn:
                # self.Debug (4, "Closing DB connection")
                self.conn.close()
            if self.engine:
                # self.Debug (4, "Closing DB engine")
                self.engine.dispose()
                # Put this here so it logs to syslog and file before closing those handlers
            self.debug(f"{lp} Closing all log handlers NOW", caller=caller)
            if not g.logger:
                print(f"{lp} Closing all log handlers NOW")
            if self.syslog:
                # self.Debug (4, "Closing syslog connection")
                self.syslog.closelog()
            if self.log_file_handler:
                # self.Debug (4, "Closing log file handle")
                self.log_file_handler.close()
                self.log_file_handler = None
            self.config['dump_console'] = True
            # Restore stdout and stderr
            if sys.stdout != sys.__stdout__:
                sys.stdout = sys.__stdout__
            if sys.stderr != sys.__stderr__:
                sys.stderr = sys.__stderr__

    def _format_string(self, message='', level='ERR'):
        log_string = '{level} [{pname}] [{message}]'.format(level=level, pname=self.process_name, message=message)
        return (log_string)

    def time_format(self, dt_form):
        if len(str(float(f"{dt_form.microsecond / 1e6}")).split(".")) > 1:
            micro_sec = str(float(f"{dt_form.microsecond / 1e6}")).split(".")[1]
        else:
            micro_sec = str(float(f"{dt_form.microsecond / 1e6}")).split(".")[0]
        return "%s.%s" % (
            dt_form.strftime('%m/%d/%y %H:%M:%S'),
            micro_sec
        )

    def _log(self, **kwargs):
        blank, caller, nl, level, tight, debug_level, message = None, None, None, 'DBG', False, 1, None
        dt = None
        display_level = None
        fnfl = None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v
            elif k == 'level':
                level = v
            elif k == 'message':
                message = v
            elif k == 'debug_level':
                debug_level = v
            elif k == 'blank':
                blank = v
            elif k == 'timestamp':
                dt = v
            elif k == 'display_level':
                display_level = v
            elif k == 'filename' and v is not None:
                fnfl = f"{v}:{kwargs['lineno']}"
        file_log_string = ''
        if not dt:
            dt = self.time_format(datetime.now())
        # first stack element will be the wrapper log function
        # second stack element will be the actual calling function or maybe class wrapper?
        # third will be actual func?
        # print (len(stack()))
        if not caller or caller and fnfl:  # if caller was not passed we create it here, meaning in logs this modules name and line no will show up
            idx = min(len(stack()), 2)  # in the case of someone calls this directly
            caller = getframeinfo(stack()[idx][0])
        # print ('CALLER INFO --> FILE: {} LINE: {}'.format(caller.filename, caller.lineno))
        # If we are debug logging, show level too
        if not display_level:
            display_level = level
        file_log_string = '{level} [{pname}] [{msg}]'.format(level=display_level, pname=self.process_name, msg=message)

        if level == 'DBG':
            display_level = f'DBG{debug_level}'

        # write to syslog
        if levels[level] <= self.config['log_level_syslog']:
            file_log_string = '{level} [{pname}] [{msg}]'.format(level=display_level, pname=self.process_name,
                                                                 msg=message)
            self.syslog.syslog(priorities[level], file_log_string)

        component = self.process_name
        serverid = self.config.get('server_id')
        pid = self.pid
        level_ = levels[level]
        code = level
        line = caller.lineno
        # write to db only if ZM installed
        send_to_db = True
        if levels[level] <= self.config['log_level_db'] and zm_inst:
            if not self.sql_connected:
                self.syslog.syslog(syslog.LOG_INFO, self._format_string("Connecting to ZoneMinder SQL DB"))
                if not self._db_reconnect():
                    self.syslog.syslog(syslog.LOG_ERR, self._format_string("Reconnecting failed, not writing to ZM DB"))
                    send_to_db = False
            if send_to_db:
                try:
                    cmd = self.log_table.insert().values(TimeKey=time.time(), Component=component, ServerId=serverid,
                                                         Pid=pid, Level=level_, Code=code, Message=message,
                                                         File=os.path.split(caller.filename)[1], Line=line)
                    self.conn.execute(cmd)
                except SQLAlchemyError as e:
                    self.sql_connected = False
                    self.syslog.syslog(syslog.LOG_ERR, self._format_string("Error writing to DB:" + str(e)))

        print_log_string = None
        if levels[level] <= self.config['log_level_file']:
            if not fnfl:
                fnfl = f"{os.path.split(caller.filename)[1].split('.')[0]}:{caller.lineno}"
            print_log_string = '{timestamp} {pname}[{pid}] {level} {fnfl}->[{msg}]'.format(timestamp=dt,
                                                                                           level=display_level,
                                                                                           pname=self.process_name,
                                                                                           pid=pid, msg=message,
                                                                                           fnfl=fnfl)
            file_log_string = '{timestamp} {pname}[{pid}] {level} {fnfl} [{msg}]\n'.format(timestamp=dt,
                                                                                           level=display_level,
                                                                                           pname=self.process_name,
                                                                                           pid=pid, msg=message,
                                                                                           fnfl=fnfl)

            if self.log_file_handler:
                try:
                    self.log_file_handler.write(file_log_string)  # write to file
                    self.log_file_handler.flush()
                except Exception as e:
                    sys.stdout = sys.__stdout__
                    print(f"'{file_log_string}' is not available to be written to, trying to create a new log file")
                    # make log file
                    uid = pwd.getpwnam(self.config['webuser']).pw_uid
                    gid = grp.getgrnam(self.config['webgroup']).gr_gid
                    if self.config['log_level_file'] > levels['OFF']:
                        n = os.path.split
                        self.log_filename = f"{self.config['logpath']}/{n(self.process_name)[1].split('.')[0]}.log"
                        # print ('WRITING TO {}'.format(self.log_fname))
                        try:
                            log_file = Path(self.log_filename)
                            log_file.touch(exist_ok=True)
                            self.log_file_handler = log_file.open('a')
                            os.chown(self.log_filename, uid, gid)  # proper permissions
                        except Exception as e:
                            self.syslog.syslog(
                                syslog.LOG_ERR,
                                self._format_string(
                                    f"Error for user: {getuser()} while creating and changing permissions of "
                                    f"log file -> {str(e)}"
                                )
                            )
                            self.log_file_handler = None
                        else:
                            self.log_file_handler.write(file_log_string)  # write to file after re creating log file
                            self.log_file_handler.flush()

        if self.config['dump_console']:
            if tight:
                if nl:
                    print(f"\n{print_log_string}")
                else:
                    print(print_log_string)
            else:
                print(f"\n{print_log_string}")

    def info(self, message, **kwargs):
        tight = False
        caller, nl, level = None, None, None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v
        self._log(level='INF', message=message, caller=caller, tight=tight, nl=nl)

    def debug(self, *args, **kwargs):
        tight = False
        caller, nl, level, message = None, None, None, None
        level = 1
        ts = None
        message = None
        lineno = None
        display_level = None
        filename = None
        if isinstance(args[0], dict):
            a = args[0]
            ts = a['timestamp']
            lineno = a['lineno']
            display_level = a['display_level']
            filename = a['filename']
            message = a['message']

        elif len(args) == 1:
            message = args[0]
        elif len(args) == 2:
            level = args[0]
            message = args[1]

        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v
        if not caller:
            idx = min(len(stack()), 1)  # in the case of someone calls this directly
            caller = getframeinfo(stack()[idx][0])

        target = self.config['log_debug_target']
        if target and not self.config['dump_console']:
            targets = [x.strip().lstrip('_') for x in target.split('|')]
            # if current name does not fall into debug targets don't log
            if not any(map(self.process_name.startswith, targets)):
                return
        if self.config['log_debug'] and level <= self.config['log_level_debug']:
            self._log(
                level='DBG',
                message=message,
                caller=caller,
                debug_level=level,
                tight=tight,
                nl=nl,
                timestamp=ts,
                display_level=display_level,
                filename=filename,
                lineno=lineno
            )

    def warning(self, message, **kwargs):
        tight = False
        caller, nl, level = None, None, None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v

        self._log(level='WAR', message=message, caller=caller, tight=tight, nl=nl)

    def error(self, message, **kwargs):
        tight = False
        caller, nl, level = None, None, None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v

        self._log(level='ERR', message=message, caller=caller, tight=tight, nl=nl)

    def fatal(self, message, **kwargs):
        tight = False
        caller, nl, level = None, None, None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v

        self._log(level='FAT', message=message, caller=caller, tight=tight, nl=nl)
        self.log_close()
        exit(-1)

    def panic(self, message, **kwargs):
        tight = False
        caller, nl, level = None, None, None
        for k, v in kwargs.items():
            if k == 'caller':
                caller = v
            elif k == 'tight':
                tight = v
            elif k == 'nl':
                nl = v
        self._log(level='PNC', message=message, caller=caller, tight=tight, nl=nl)
        self.log_close()
        exit(-1)
