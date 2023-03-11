"""
This file contains the model architecture for our feedforward neural network
"""

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization


class DenseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, kernel_regularizer = regularizers.l2(1e-5), activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(32, kernel_regularizer = regularizers.l2(1e-5), activation = 'relu')
        self.batch_normalize = BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(32, kernel_regularizer = regularizers.l2(1e-5), activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(1, activation = None)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.batch_normalize(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return x
