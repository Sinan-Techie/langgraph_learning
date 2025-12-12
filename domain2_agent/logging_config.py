
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='app.log'):
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

setup_logging(log_file="domain2_agent.log")
logger = logging.getLogger(__name__)