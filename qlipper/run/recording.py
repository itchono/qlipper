import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def temp_log_to_file(log_file: Path):
    """
    Temporarily enable logging to a file, then remove the handler when
    the context manager exits.

    Parameters
    ----------
    log_file : Path
        Path to the log file.
    """
    # add handler to root logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(file_handler)

    try:
        yield

    finally:
        logging.getLogger().removeHandler(file_handler)
