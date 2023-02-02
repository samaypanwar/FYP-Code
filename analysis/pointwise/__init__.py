import tensorflow as tf

from analysis.utils import begin_logging

tf.keras.backend.set_floatx('float64');

logger, training_logger, calibration_logger = begin_logging()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# TODO: Calibrate models using real world data
# TODO: Fix as much linting errors as possible
