import tensorflow as tf
from tensorflow.keras import layers

class GaussianDiffusion(layers.Layer):
    def __init__(self, **kwargs):
        super(GaussianDiffusion, self).__init__(**kwargs)
        self.std_dev = tf.Variable(initial_value=0.1, trainable=True, dtype=tf.float32)

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.std_dev)
            return inputs + noise
        else:
            return inputs

    def get_config(self):
        config = super(GaussianDiffusion, self).get_config()
        config.update({"std_dev": self.std_dev.numpy()})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

