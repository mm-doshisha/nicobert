import datetime
import logging
import logging.config
import time

import yaml

initialized_logger = {}


def get_logger_from_yaml(file_path):
    with open(file_path) as f:
        data = yaml.safe_load(f.read())
        logging.config.dictConfig(data)
    logger = logging.getLogger("nicobert")
    return logger


def get_root_logger(logger_name="nicobert", log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger

    format_str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)

    logger.propagate = False

    if log_file is not None:
        logger.setLevel(log_level)

        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    initialized_logger[logger_name] = True
    return logger
