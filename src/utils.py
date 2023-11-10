import json
import logging
import os
import sys
import time
from datetime import datetime
from functools import wraps


# def setup_logger(name='', level=logging.INFO, log_file='active_learning_run.log'):
def setup_logger(name='', level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        tmp_log_file = os.path.join('/tmp', log_file)
        file_handler = logging.FileHandler(tmp_log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger, tmp_log_file

    return logger


def measure_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        minutes, seconds = divmod(duration, 60)
        formatted_time = f"{int(minutes)}m{int(seconds)}s"

        return result, formatted_time

    return wrapper


def format_with_border(message, total_length=100):
    border_length = int((total_length - len(message) - 2) / 2)
    border = '=' * border_length

    return f'{border} {message} {border}'
