import tensorflow as tf


class DenseModel(tf.keras.Model):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(30, activation = 'elu');
        self.dense2 = tf.keras.layers.Dense(30, activation = 'elu');
        self.dense3 = tf.keras.layers.Dense(30, activation = 'elu')
        self.dense4 = tf.keras.layers.Dense(1, activation = None)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x
