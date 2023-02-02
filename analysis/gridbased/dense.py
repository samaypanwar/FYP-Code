import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

from hyperparameters import number_of_coupon_rates, number_of_maturities


class DenseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, kernel_regularizer = regularizers.l2(1e-5), activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(32, kernel_regularizer = regularizers.l2(1e-5), activation = 'relu')
        self.batch_normalize = BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(32, kernel_regularizer = regularizers.l2(1e-5), activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(number_of_maturities * number_of_coupon_rates, activation = None)
        self.reshape = tf.keras.layers.Reshape([number_of_maturities, number_of_coupon_rates])

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.batch_normalize(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.reshape(x)

        return x
