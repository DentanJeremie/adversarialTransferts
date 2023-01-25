import logging
import sys

from src.utils.pathtools import project

logger = logging.root
logFormatter = logging.Formatter('{relativeCreated:12.0f}ms {levelname:5s} [{filename}] {message:s}', style='{')
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(project.get_log_file('full'))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

sum_fileHandler = logging.FileHandler(project.get_log_file('summary'))
sum_fileHandler.setFormatter(logFormatter)
sum_fileHandler.setLevel(logging.INFO)
logger.addHandler(sum_fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.INFO)
logger.addHandler(consoleHandler)

# Silent unuseful log
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', message=r'This implementation of AdamW*')
warnings.filterwarnings('ignore', message=r'nn.glob.global_sort_pool')
from matplotlib import pyplot as plt
plt.set_loglevel("warning")
