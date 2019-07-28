"""
Base
======
All classes derive from this Base class.
It implements some common functions that apply across all.
For now, this basically holds a pointer to the logging function to invoke
to log messages including a simple console based print function if none is provided

"""

class Base:
    def __init__(self, logger):
        if not logger:
            self.logger = SimpleLog()
            self.logger.Info ('Using simple log output (default)')
            
        else:
            self.logger = logger

class SimpleLog:
    ' console based logging function that is used if no logging handler is passed'
    def __init__(self):
        pass

    def Debug (self,level, message, caller=None):
        print ('[DEBUG {}] {}'.format(level, message))

    def Info (self,message, caller=None):
        print ('[INFO] {}'.format( message))

    def Error (self,message, caller=None):
        print ('[ERROR] {}'.format(message))

    def Fatal (self,message, caller=None):
        print ('[FATAL] {}'.format(message))
        exit(-1)

    def Panic (self,message, caller=None):
        print ('[PANIC] {}'.format(message))
        exit(-2)