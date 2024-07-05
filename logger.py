import logging
import logging.handlers as handlers
import os

logger = None
default_logger_name = 'pole'
log_dir = './log'

def make_logger(name=None):
    logger = logging.getLogger(name)
    filename = default_logger_name + '.log'
    if name is not None:
        filename = name + '.log'
    filename = os.path.join(log_dir, filename)

    os.makedirs(log_dir, exist_ok=True)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    # file_handler = logging.FileHandler(filename=filename)
    file_handler = handlers.TimedRotatingFileHandler(filename=filename, when='D', interval=1, backupCount=7)

    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

def get_logger(name=None):
    global logger

    if logger is None:
        logger = make_logger(name)

    return logger
