import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # get the current working directory getcwd() and join it with logs folder
os.makedirs(log_path, exist_ok=True) # create the logs folder if it does not exist

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s", #{%(pathname)s:}
    level=logging.INFO # level can be changed to DEBUG, INFO, WARNING, ERROR, CRITICAL
)