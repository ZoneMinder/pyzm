"""
Base
======
All classes derive from this Base class.
It implements some common functions that apply across all.
For now, this basically holds a pointer to the logging function to invoke
to log messages including a simple console based print function if none is provided

"""

from datetime import datetime

class Base:
    def __init__(self, logger):
        #print ('core:logger is {}'.format(logger))
        if not logger:
            self.logger = SimpleLog()
            self.logger.Info ('Using simple log output (default)')
            
        else:
            self.logger = logger

class SimpleLog:
    ' console based logging function that is used if no logging handler is passed'
    def __init__(self):
        self.dtformat = "%b %d %Y %H:%M:%S.%f"

    def Debug (self,level, message, caller=None):
        dt = datetime.now().strftime(self.dtformat)

        print ('{} [DBG {}] {}'.format(dt, level, message))

    def Info (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [INF] {}'.format( dt, message))

    def Warning (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{}  [WAR] {}'.format( dt, message))

    def Error (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [ERR] {}'.format(dt, message))

    def Fatal (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [FAT] {}'.format(dt, message))
        exit(-1)

    def Panic (self,message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print ('{} [PNC] {}'.format(dt, message))
        exit(-2)
