import logging
from src.ml.logger import setup_logger


def test_setup_logger():
    logger = setup_logger(__name__)
    assert type(logger) == logging.Logger

    logger = setup_logger(__name__)
    assert len(logger.handlers) == 1
