class QuaternionRotation(layers.Layer):
    def __init__(self, **kwargs):
        super(QuaternionRotation, self).__init__(**kwargs)

    def build(self, input_shape):
        self.quaternion_w = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=True,
            name="quaternion_w"
        )
        self.quaternion_x = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=True,
            name="quaternion_x"
        )
        self.quaternion_y = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=True,
            name="quaternion_y"
        )
        self.quaternion_z = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=True,
            name="quaternion_z"
        )
        super(QuaternionRotation, self).build(input_shape)

    def call(self, x):
        if training:
            quaternion = tf.stack([self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z], axis=-1)
            normalized_quaternion = tf.linalg.l2_normalize(quaternion)
            rotated_x = quaternion.rotate(x, normalized_quaternion)
            return rotated_x
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

