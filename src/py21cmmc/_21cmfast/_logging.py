import logging
import os
import sys
from multiprocessing import current_process


class PIDFormatter(logging.Formatter):
    _mylogger = logging.getLogger("21CMMC") # really bad hack

    def format(self, record):
        fmt = "{asctime} | {levelname} |"

        if self._mylogger.level <= logging.DEBUG:
            fmt += " {filename}::{funcName}() |"

        if current_process().name != "MainProcess":
            fmt += " pid={process} |"

        self._style = logging.StrFormatStyle(fmt + " {message}")

        return logging.Formatter.format(self, record)

def configure_logging():
    # logging.basicConfig()
    hdlr = logging.StreamHandler(sys.stderr)
    hdlr.setFormatter(PIDFormatter())
    logger = logging.getLogger("21CMMC")
    logger.addHandler(hdlr)
