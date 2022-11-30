import tensorflow as tf

from analysis.gridbased import N1, N2


class DenseModel(tf.keras.Model):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(30, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(30, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(30, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(N1 * N2, activation = None)
        self.reshape = tf.keras.layers.Reshape([N1, N2])

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.reshape(x)
        return x
