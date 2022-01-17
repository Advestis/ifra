import logging.config
from transparentpath import Path
import os


def full_stack():
    import traceback
    import sys
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
        stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr


def setup_logger(default_path="logging.json", default_level=logging.INFO, env_key="LOG_CFG", fs="local", **kwargs):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    path = Path(path, fs=fs, **kwargs)
    if path.is_file():
        config = path.read()
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
