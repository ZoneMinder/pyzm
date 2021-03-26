"""
ZMLog
=======
Implements a python implementation of ZoneMinder's logging system
You could use this to connect it to the APIs if you want

"""

from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import select
from sqlalchemy import or_
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

import configparser
import glob,os,psutil
import syslog
from inspect import getframeinfo,stack
import time
import pwd,grp
import datetime
import signal
import sys
import pyzm.helpers.globals as g


pid = None
process_name = None
inited = False
config={}
engine = None
conn = None
connected = False
config_table = None
log_table = None
meta = None
log_fname = None
log_fhandle = None
cstr = None
log_reload_needed = False

logger_name = None 
logger_overrides = {}


connected = False
levels = {
    'DBG':1,
    'INF':0,
    'WAR':-1,
    'ERR':-2,
    'FAT':-3,
    'PNC':-4,
    'OFF':-5
    }

priorities = {
        'DBG':syslog.LOG_DEBUG,
        'INF':syslog.LOG_INFO,
        'WAR':syslog.LOG_WARNING,
        'ERR':syslog.LOG_ERR,
        'FAT':syslog.LOG_ERR,
        'PNC':syslog.LOG_ERR
    }

def init(name=None, override={}):
    """Initializes the ZM logging system. It follows ZM logging principles and ties into appropriate ZM logging levels. Like the rest of ZM, it can write to files, syslog and the ZM DB.

    To make it simpler to override, you can pass various options in the override dict. When passed, they will override any ZM setting
    
    Args:
        name (string, optional): Name to be used while writing logs. If not specified, it will use the process name. Defaults to None.
        override (dict, optional): Various parameters that can supercede standard ZM logs. Defaults to {}. The parameters that can be overriden are::

            {
                'dump_console': False,
                'conf_path': '/etc/zm',
                'user' : None,
                'password' : None,
                'host' : None,
                'webuser': 'www-data',
                'webgroup': 'www-data',
                'dbname' : None,
                'logpath' : '/var/log',
                'log_level_syslog' : 0,
                'log_level_file' : 0,
                'log_level_db' : 0,
                'log_debug' : 0,
                'log_level_debug' : 1,
                'log_debug_target' : '',
                'log_debug_file' :'',
                'server_id': 0,
                'driver': 'mysql+mysqlconnector'
            }
    """
    global logger, pid, process_name, inited, config, engine, conn, cstr, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    global logger_name, logger_overrides
    if inited:
        Debug (1, "Logs already inited")
        return

    inited = True
    logger_name = name 
    logger_overrides = override
    pid =  os.getpid()
    process_name = name or psutil.Process(pid).name()
    syslog.openlog(logoption=syslog.LOG_PID)

    config = {
        'conf_path': '/etc/zm',
        'user' : None,
        'password' : None,
        'host' : None,
        'webuser': 'www-data',
        'webgroup': 'www-data',
        'dbname' : None,
        'logpath' : '/var/log',
        'log_level_syslog' : 0,
        'log_level_file' : 0,
        'log_level_db' : 0,
        'log_debug' : 0,
        'log_level_debug' : 1,
        'log_debug_target' : '',
        'log_debug_file' :'',
        'server_id': 0,
        'driver': 'mysql+mysqlconnector',
        'dump_console': False
    }

    # Round 1 of overrides, before we read params from DB
        # Override with locals if present
    for key in config:
        if key in override:
            config[key] = override[key]
    


    # read all config files in order
    files=[]
    for f in glob.glob(config['conf_path']+'/conf.d/*.conf'):
        files.append(f)
    files.sort()
    files.insert(0,config['conf_path']+'/zm.conf')
    config_file = configparser.ConfigParser(interpolation=None, inline_comment_prefixes='#')
    for f in files:
        with open(f) as s:
            #print ('reading {}'.format(f))
            config_file.read_string('[zm_root]\n' + s.read())
            s.close()

    # config_file will now contained merged data
    conf_data=config_file['zm_root']

    config['user'] = conf_data.get('ZM_DB_USER', 'zmuser')
    config['password'] = conf_data.get('ZM_DB_PASS', 'zmpass')
    config['webuser'] = conf_data.get('ZM_WEB_USER', 'www-data')
    config['webgroup'] = conf_data.get('ZM_WEB_GROUP', 'www-data')
    config['host'] = conf_data.get('ZM_DB_HOST', 'localhost')
    config['dbname'] = conf_data.get('ZM_DB_NAME', 'zm')
    config['logpath'] =  config_file['zm_root']['ZM_PATH_LOGS']

    cstr = config['driver']+'://{}:{}@{}/{}'.format(config['user'],
        config['password'],config['host'],config['dbname'])

    try:
        engine = create_engine(cstr, pool_recycle=3600)
        conn = engine.connect()
        connected = True
    except SQLAlchemyError as e:
        connected = False
        conn = None
        engine = None
        syslog.syslog (syslog.LOG_ERR, _format_string("Turning DB logging off. Could not connect to DB, message was:" + str(e)))
        config['log_level_db'] = levels['OFF']
        
    else:
        meta = MetaData(engine,reflect=True)
        config_table = meta.tables['Config']
        log_table = meta.tables['Logs']

        select_st = select([config_table.c.Name, config_table.c.Value]).where(
                or_(config_table.c.Name=='ZM_LOG_LEVEL_SYSLOG',
                    config_table.c.Name=='ZM_LOG_LEVEL_FILE',
                    config_table.c.Name=='ZM_LOG_LEVEL_DATABASE',
                    config_table.c.Name=='ZM_LOG_DEBUG',
                    config_table.c.Name=='ZM_LOG_DEBUG_LEVEL',
                    config_table.c.Name=='ZM_LOG_DEBUG_FILE',
                    config_table.c.Name=='ZM_LOG_DEBUG_TARGET',
                    config_table.c.Name=='ZM_SERVER_ID',
                    ))
        resultproxy = conn.execute(select_st)
        db_vals = {row[0]:row[1] for row in resultproxy}
        config['log_level_syslog'] = int(db_vals['ZM_LOG_LEVEL_SYSLOG']) 
        config['log_level_file'] = int(db_vals['ZM_LOG_LEVEL_FILE'])
        config['log_level_db'] = int(db_vals['ZM_LOG_LEVEL_DATABASE'])
        config['log_debug'] = int(db_vals['ZM_LOG_DEBUG'])
        config['log_level_debug'] = int(db_vals['ZM_LOG_DEBUG_LEVEL'])
        config['log_debug_file'] = db_vals['ZM_LOG_DEBUG_FILE']
        config['log_debug_target'] = db_vals['ZM_LOG_DEBUG_TARGET']
        config['server_id'] = db_vals.get('ZM_SERVER_ID',0)
    # Round 2 of overrides, after DB data is read
    # Override with locals if present
    for key in config:
        if key in override:
            config[key] = override[key]

    log_fname = None
    log_fhandle = None

    if config['log_level_file'] > levels['OFF']:
        
        n = os.path.split(process_name)[1].split('.')[0]
        log_fname = config['logpath']+'/'+n+'.log' 
        try:
            log_fhandle = open (log_fname,'a')
            uid = pwd.getpwnam(config['webuser']).pw_uid
            gid = grp.getgrnam(config['webgroup']).gr_gid
            os.chown(log_fname, uid,gid)
        except OSError as e:
            syslog.syslog (syslog.LOG_ERR, _format_string("Error opening file log:" + str(e)))
            log_fhandle = None
    try:
        Info('Setting up signal handler for logs')
        signal.signal(signal.SIGHUP, sig_log_rot)
        signal.signal(signal.SIGINT, sig_intr)

    except Exception as e:
        Error('Error setting up signal handler: {}'.format(e))

    g.logger = sys.modules[__name__]
    #print (sys.modules[__name__])
    Info ('Switching global logger to ZMLog')


def sig_log_rot(sig,frame):
    #time.sleep(3) # do we need this?
    global log_reload_needed, inited, logger_name, logger_overrides

    log_reload_needed = True
    if inited:
        close()
        init(name=logger_name, override=logger_overrides)
        Info ('Ready after log re-init')
    log_reload_needed = False
    Info('Got HUP signal:{}, re-inited logs'.format(sig))
    
def sig_intr(sig,frame):
    Info ('Got interrupt, exiting')
    close()
    exit(0)

def set_level(level):
    pass

def get_config():
    """Returns configuration of ZM logger
    
    Returns:
        dict: config params
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    return config

def _db_reconnect():
    """Invoked by the logger if disconnection occurs
    
    Returns:
        boolean: True if reconnected
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    try:
        conn.close()
    except:
        pass

    try:
        engine = create_engine(cstr, pool_recycle=3600)
        conn = engine.connect()
        #inspector = inspect(engine)
        #print(inspector.get_columns('Config'))
        meta = MetaData(engine,reflect=True)
        config_table = meta.tables['Config']
        log_table = meta.tables['Logs']
        message = 'reconnecting to Database...'
        log_string = '{level} [{pname}] [{msg}]'.format(level='INF', pname=process_name, msg=message)
        syslog.syslog (syslog.LOG_INFO, log_string)
    except SQLAlchemyError as e:
        connected = False
        syslog.syslog (syslog.LOG_ERR, _format_string("Turning off DB logging due to error received:" + str(e)))
        return False
    else:
        connected = True
        return True
        

def close():
    """Closes all handles. Invoke this before you exit your app
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    if conn: 
        #Debug (4, "Closing DB connection")
        conn.close()
    if engine: 
        #Debug (4, "Closing DB engine")
        engine.dispose()
    #Debug (4, "Closing syslog connection")
    syslog.closelog()
    if (log_fhandle): 
        #Debug (4, "Closing log file handle")
        log_fhandle.close()
    inited = False

def _format_string(message='', level='ERR'):
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    log_string = '{level} [{pname}] [{message}]'.format(level=level, pname=process_name, message=message)
    return (log_string)

def _log(level='DBG', message='', caller=None, debug_level=1):
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    log_string=''
    if not inited:
        raise ValueError ('Logs not initialized')
    # first stack element will be the wrapper log function
    # second stack element will be the actual calling function
    #print (len(stack()))
    if not caller:
        idx = min(len(stack()), 2) #incase someone calls this directly
        caller = getframeinfo(stack()[idx][0])
        #print ('Called from {}:{}'.format(caller.filename, caller.lineno))

    # If we are debug logging, show level too
    disp_level=level
    if level=='DBG':
        disp_level = f'DBG{debug_level}'
    log_string = '{level} [{pname}] [{msg}]'.format(level=disp_level, pname=process_name, msg=message)
    # write to syslog
   
    if levels[level] <= config['log_level_syslog']:
        syslog.syslog (priorities[level], log_string)

    # write to db
    if levels[level] <= config['log_level_db']:
        if not connected:
            syslog.syslog (syslog.LOG_INFO, _format_string("Trying to reconnect"))
            if not _db_reconnect():
                syslog.syslog (syslog.LOG_ERR, _format_string("reconnection failed, not writing to DB"))
            return False

        log_string = '{level} [{pname}] [{msg}]'.format(level=disp_level, pname=process_name, msg=message)
        component = process_name
        serverid = config['server_id']
        pid = pid
        l = levels[level]
        code = level
        line = caller.lineno

        try:
            cmd = log_table.insert().values(TimeKey=time.time(), Component=component, ServerId=serverid, Pid=pid, Level=l, Code=code, Message=message,File=os.path.split(caller.filename)[1], Line=line)
            conn.execute(cmd)
        except SQLAlchemyError as e:
            connected = False
            syslog.syslog (syslog.LOG_ERR, _format_string("Error writing to DB:" + str(e)))
 
    # write to file components
    if levels[level] <= config['log_level_file']:
        timestamp = datetime.datetime.now().strftime('%x %H:%M:%S')
        # 07/15/19 10:10:14.050651 zmc_m8[4128].INF-zm_monitor.cpp/2516 [Driveway: images:218900 - Capturing at 3.70 fps, capturing bandwidth 98350bytes/sec]
        fnfl ='{}:{}'.format(os.path.split(caller.filename)[1], caller.lineno)
        log_string = '{timestamp} {pname}[{pid}] {level} {fnfl} [{msg}]\n'.format(timestamp=timestamp, level=disp_level, pname=process_name, pid=pid, msg=message, fnfl=fnfl)
        if log_fhandle: 
            log_fhandle.write(log_string)
            log_fhandle.flush()

    if config['dump_console']:
        print (log_string)

def Info(message=None,caller=None):
    """Info level ZM message
    
    Args:
        message (string): Message to log
        caller (stack frame info, optional): Used to log caller id/line #. Picked up automatically if none. Defaults to None.
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    _log('INF',message,caller)

def Debug(level=1, message=None,caller=None):
    """Debug level ZM message
    
    Args:
        level (int): ZM Debug level
        message (string): Message to log
        caller (stack frame info, optional): Used to log caller id/line #. Picked up automatically if none. Defaults to None.
    """

    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    target = config['log_debug_target']

    if target and not config['dump_console']:
        targets = [x.strip().lstrip('_') for x in target.split('|')]
        # if current name does not fall into debug targets don't log
        if not any(map(process_name.startswith, targets)):
            return

    
    if config['log_debug'] and level <= config['log_level_debug']:
        _log('DBG', message,caller, level)

def Warning(message=None,caller=None):
    """Warning level ZM message
    
    Args:
        message (string): Message to log
        caller (stack frame info, optional): Used to log caller id/line #. Picked up automatically if none. Defaults to None.
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    _log('WAR',message,caller)

def Error(message=None,caller=None):
    """Error level ZM message
    
    Args:
        message (string): Message to log
        caller (stack frame info, optional): Used to log caller id/line #. Picked up automatically if none. Defaults to None.
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    _log('ERR',message,caller)
    
def Fatal(message=None,caller=None):
    """Fatal level ZM message. Quits after logging
    
    Args:
        message (string): Message to log
        caller (stack frame info, optional): Used to log caller id/line #. Picked up automatically if none. Defaults to None.
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    _log('FAT',message,caller)
    close()
    exit(-1)

def Panic(message=None,caller=None):
    """Panic level ZM message. Quits after logging
    
    Args:
        message (string): Message to log
        caller (stack frame info, optional): Used to log caller id/line #. Picked up automatically if none. Defaults to None.
    """
    global logger, pid, process_name, inited, config, engine, conn, connected, levels, priorities, config_table, log_table, meta, log_fname, log_fhandle
    _log('PNC',message,caller)
    close()
    exit(-1)


