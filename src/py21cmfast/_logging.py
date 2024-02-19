"""Configure logging for py21cmfast.

Significantly, adds a new formatter which prepends the PID of the logging process to
any output. This is helpful when running multiple threads in MPI.
"""

import logging
import sys
from multiprocessing import current_process


class PIDFormatter(logging.Formatter):
    """Logging formatter which prepends the PID of the logging process to any output."""

    _mylogger = logging.getLogger("21cmFAST")  # really bad hack

    def format(self, record):  # noqa
        """Set the format of the log."""
        fmt = "{asctime} | {levelname} |"

        if self._mylogger.level <= logging.DEBUG:
            fmt += " {filename}::{funcName}() |"

        if current_process().name != "MainProcess":
            fmt += " pid={process} |"

        self._style = logging.StrFormatStyle(fmt + " {message}")

        return logging.Formatter.format(self, record)


def configure_logging():
    """Configure logging for the '21cmFAST' logger."""
    hdlr = logging.StreamHandler(sys.stderr)
    hdlr.setFormatter(PIDFormatter())
    logger = logging.getLogger("21cmFAST")
    logger.addHandler(hdlr)
