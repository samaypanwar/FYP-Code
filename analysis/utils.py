import logging
import os
import sys
from stat import S_IREAD, S_IWUSR  # Need to add this import to the ones above

# ADD YOUR PATH HERE
path = "/Users/samaypanwar/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/Uni/FYP/FYP-Code"
os.chdir(path)


def begin_logging():
    """This function takes care of the logging configuration"""

    os.chmod('log.log', S_IWUSR | S_IREAD)  # This makes the file read/write for the owner
    os.chmod('training.log', S_IWUSR | S_IREAD)  # This makes the file read/write for the owner
    os.chmod('calibration.log', S_IWUSR | S_IREAD)  # This makes the file read/write for the owner

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # our first handler is a console handler
    console_handler = logging.StreamHandler(stream = sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler_format = '%(asctime)s | %(levelname)s: | %(filename)s | %(funcName)s | %(lineno)d: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)


    # the second handler is a file handler
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.INFO)
    file_handler_format = '%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    logger.propagate = False

    # the third handler is a file handler for training epochs
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.DEBUG)

    training_handler = logging.FileHandler('training.log')
    training_handler.setLevel(logging.DEBUG)
    training_handler_format = '%(asctime)s | %(levelname)s | %(message)s'
    training_handler.setFormatter(logging.Formatter(training_handler_format))
    training_logger.addHandler(training_handler)
    # training_logger.addHandler(console_handler)

    training_logger.propagate = False

    # the third handler is a file handler for calibration epochs
    calibration_logger = logging.getLogger('calibration')
    calibration_logger.setLevel(logging.DEBUG)

    calibration_handler = logging.FileHandler('calibration.log')
    calibration_handler.setLevel(logging.DEBUG)
    calibration_handler_format = '%(asctime)s | %(levelname)s | %(message)s'
    calibration_handler.setFormatter(logging.Formatter(calibration_handler_format))
    calibration_logger.addHandler(calibration_handler)

    calibration_logger.propagate = False

    return logger, training_logger, calibration_logger
