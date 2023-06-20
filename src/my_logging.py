import os
import sys

import datetime as dt
import logging

INFO_LOGGER = None
EXC_LOGGER = None

# TODO: Add single method to log everything
# E.g.: log(module:str, msg:str, exc: Exception = None) -> If Exception is not none, log to info as error and traceback to exception


def setup_loggers(root: str = None, info: bool = True, exception: bool = True):

    if root is None:
        root = os.path.basename(sys.argv[0])

    if info:
        setup_info_logger(root)

    if exception:
        setup_exception_logger(root)


def setup_info_logger(root:str):
    global INFO_LOGGER

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # logging.basicConfig(filename=f'output\\{now}.log',
    #                 level=logging.DEBUG,
    #                 format='%(asctime)s;%(levelname)s:%(message)s',
    #                 datefmt='%Y-%m-%d %H:%M:%S')

    output_file = f"output\\{root}_{now}_INFO.log"
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s\n" "->:%(message)s \n"
    )

    logger = logging.getLogger("info")
    handler = logging.FileHandler(output_file)
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    INFO_LOGGER = logger
    return INFO_LOGGER


def setup_exception_logger(root:str):
    global EXC_LOGGER

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # logging.basicConfig(filename=f'output\\{now}.log',
    #                 level=logging.DEBUG,
    #                 format='%(asctime)s;%(levelname)s:%(message)s',
    #                 datefmt='%Y-%m-%d %H:%M:%S')

    output_file = f"output\\{root}_{now}_EXC.log"
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s\n" "->\n" "->%(message)s \n\n"
    )

    logger = logging.getLogger("exception")
    handler = logging.FileHandler(output_file)
    handler.setFormatter(formatter)
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)

    EXC_LOGGER = logger
    return EXC_LOGGER


# TODO: Test
class MyLogging:
    def __init__(self, info: bool = True, exception: bool = True) -> None:
        self.INFO_LOGGER: logging.Logger = self.setup_info_logger() if info else None
        self.EXC_LOGGER: logging.Logger = (
            self.setup_exception_logger() if exception else None
        )

    def log(self, module: str, msg: str, exc: Exception = None):

        assert (
            msg is not None or exc is not None
        ), "Pass a message or a Exception to log"

        if exc is not None:
            self.INFO_LOGGER.error(msg)
            self.EXC_LOGGER.exception(exc)
            return

        self.INFO_LOGGER.info(msg)

    def setup_info_logger(self):

        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = f"output\\{now}_INFO.log"
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s\n" "->%(levelname)s:%(message)s \n"
        )

        logger = logging.getLogger("info")
        handler = logging.FileHandler(output_file)
        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        INFO_LOGGER = logger
        return INFO_LOGGER

    def setup_exception_logger(self):

        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = f"output\\{now}_EXC.log"
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s\n" "->%(levelname)s:\n" "->%(message)s \n\n"
        )

        logger = logging.getLogger("exception")
        handler = logging.FileHandler(output_file)
        handler.setFormatter(formatter)
        logger.setLevel(logging.ERROR)
        logger.addHandler(handler)

        EXC_LOGGER = logger
        return EXC_LOGGER
