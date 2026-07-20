import sys
import os

class Logger:
    """
    A simple logger that redirects stdout to both terminal and a file.
    """
    def __init__(self, filename):
        # Always try to find the "real" stdout if we've already redirected it
        if hasattr(sys.stdout, 'terminal'):
            self.terminal = sys.stdout.terminal
        else:
            self.terminal = sys.stdout
            
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()
            self.log = None

def setup_logger(log_path):
    """
    Redirects sys.stdout to the specified log_path.
    Returns the Logger instance.
    """
    logger = Logger(log_path)
    sys.stdout = logger
    return logger
