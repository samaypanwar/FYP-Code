"""
This directory handles all the training/calibration of our models
It contains the following files:
    1. dense.py: contains the model architecture
    2. model_calibration.py: contains the model claibration, both the synthetic calibration and the market calibration
    3. model_training.py: contains the model training function
    4. utils.py: contains the supplementary functions such as loading weights and initialising models
"""

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

tf.random.set_seed(42)
