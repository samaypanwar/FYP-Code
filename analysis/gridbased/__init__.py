import numpy as np
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

# Size of the data
train_size = 40_000;  # Size of the training set
test_size = 4_000;  # Size of the test set

maturities = np.array([1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 1, 2, 3, 5, 10, 20, 30])

N1 = len(maturities)
