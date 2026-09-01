import logging


def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
