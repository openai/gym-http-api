import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from baselines.a2c.utils import fc, conv_to_fc
from baselines.common.distributions import make_pdtype

def cnn7(unscaled_images, **conv_kwargs):
    """
    Network 96x96:
    model/SeparableConv2d/depthwise_weights:0 (8, 8, 4, 1)
    model/SeparableConv2d/pointwise_weights:0 (1, 1, 4, 32)
    model/SeparableConv2d/biases:0 (32,)
    model/SeparableConv2d_1/depthwise_weights:0 (4, 4, 32, 1)
    model/SeparableConv2d_1/pointwise_weights:0 (1, 1, 32, 64)
    model/SeparableConv2d_1/biases:0 (64,)
    model/SeparableConv2d_2/depthwise_weights:0 (3, 3, 64, 1)
    model/SeparableConv2d_2/pointwise_weights:0 (1, 1, 64, 48)
    model/SeparableConv2d_2/biases:0 (48,)
    model/fc1/w:0 (6912, 512)
    model/fc1/b:0 (512,)
    model/v/w:0 (512, 1)
    model/v/b:0 (1,)
    model/pi/w:0 (512, 7)
    model/pi/b:0 (7,)
    Trainable variables:
    3550296
    """
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
        activ = tf.nn.relu
        h = slim.separable_conv2d(scaled_images, 32, 8, 1, 4)
        h2 = slim.separable_conv2d(h, 64, 4, 1, 2)
        h3 = slim.separable_conv2d(h2, 48, 3, 1, 1)
        h3 = conv_to_fc(h3)
        return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        self.pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = cnn7(X, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            print("Network:")
            [print(v.name, v.shape) for v in tf.trainable_variables()]
            print("Trainable variables:")
            print(np.sum([np.prod(v.get_shape()) for v in tf.trainable_variables()]))

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
