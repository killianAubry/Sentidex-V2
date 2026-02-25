import logging
import os
from pathlib import Path

# Ensure logs directory exists
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "system.log"

logger = logging.getLogger("sentidex")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Create file handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

def get_logger():
    return logger
