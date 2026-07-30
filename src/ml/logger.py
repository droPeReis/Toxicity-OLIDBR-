import os
import sys
import logging


def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger.

    Args:
    - name: The name of the logger.

    Returns:
    - logger: The logger.
    """
    log_level = os.environ.get("SM_LOG_LEVEL", logging.INFO)
    if isinstance(log_level, str) and log_level.isdigit():
        log_level = int(log_level)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Check if the logger already has a StreamHandler
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s :: %(levelname)s :: %(module)s :: %(funcName)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
