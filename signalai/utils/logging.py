import sys
import os

class Logger:
    """
    A simple logger that redirects stdout to both terminal and a file.
    """
    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
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
    Redirects sys.stdout to the specified log_path, truncating any stale
    content left over from a previous run.

    If sys.stdout is already a Logger writing to this same path (e.g.
    BaseExperiment re-initializing logging for a run directory that the
    calling script already set up), reuses it instead of reopening --
    reopening would truncate the file and discard everything already
    logged earlier in this same run.
    """
    abs_path = os.path.abspath(log_path)
    if isinstance(sys.stdout, Logger) and sys.stdout.filename == abs_path:
        return sys.stdout

    logger = Logger(log_path)
    sys.stdout = logger
    return logger
