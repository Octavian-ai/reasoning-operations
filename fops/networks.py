
import tensorflow as tf
import numpy as np
from collections import namedtuple

from .layers import *

activations = ["tanh", "relu", "sigmoid", "softmax", "selu", "linear", 'abs', 'tanh_abs']


networks = {
}

Descriptor = namedtuple('Descriptor', ['type', 'layers', 'activation'])

for activation in activations:
	for depth in range(5):

		def d(a, b, output_width):
			v = tf.concat([a,b],-1)
			for i in range(depth):
				v = layer_dense(v, output_width, activation)
			if v.shape[-1] != output_width:
				v = layer_dense(v, output_width, "linear")
			return v

		networks[Descriptor('dense', depth, activation)] = d

		def d(a, b, output_width):
			v = tf.concat([a,b],-1)
			for i in range(depth):
				v_new = layer_dense(v, output_width, activation)
				v = v + v_new
			if v.shape[-1] != output_width:
				v = layer_dense(v, output_width, "linear")
			return v

		networks[Descriptor('dense_residual', depth, activation)] = d



