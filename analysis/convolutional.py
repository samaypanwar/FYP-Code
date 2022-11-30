import tensorflow as tf

from analysis.gridbased import N1, N2


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(30, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(N1 * N2, activation = 'relu')
        self.reshape = tf.keras.layers.Reshape([N1, N2, 1])
        self.conv1 = tf.keras.layers.Conv2D(
                filters = 30,
                kernel_size = 3,
                padding = 'same',
                activation = 'relu'
                )
        self.conv2 = tf.keras.layers.Conv2D(
                filters = 1,
                kernel_size = 3,
                padding = 'same',
                activation = None
                )

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
