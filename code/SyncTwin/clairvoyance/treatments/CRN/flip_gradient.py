"""
CODE ADAPTED FROM https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import (  # type: ignore  # pylint: disable=import-error
    ops,
)


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, alpha):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * alpha]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
