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
train_size = 40000;  # Size of the training set
test_size = 4000;  # Size of the test set

l = 1 / 12  # 1 month of delivery, fixed
# NOTE: T_1 = tau, assumption for all the experiments

# TODO: Fix plot dimensions and create functions for the same if necessary
# TODO: Test out different parameter ranges and see what works best
# TODO: Change model architecture for our dense and cnn models
# TODO: Calibrate models using real world data
# TODO: Fix as much linting errors as possible