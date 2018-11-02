
import tensorflow as tf
import numpy as np
from collections import namedtuple

from .layers import *
from .datasets import bus_width

max_depth = 2
inner_width = bus_width * 2

activations = ["linear", "tanh", "relu", "sigmoid"] #, "selu", 'abs', 'tanh_abs', "softmax"]

networks = {
}

Descriptor = namedtuple('Descriptor', ['type', 'layers', 'activation'])

for depth in range(1, max_depth+1):
	for activation in activations:
		def d(a, b, output_width):
			v = tf.concat([a,b],-1)
			for i in range(depth):
				width = output_width if i == depth-1 else inner_width
				v = layer_dense(v, width, activation)

			return v

		networks[Descriptor('dense', depth, activation)] = d


def multiply(a, b, output_width):
	a = layer_dense(a, bus_width)
	b = layer_dense(b, bus_width)
	c = tf.multiply(a,b)
	return layer_dense(c, output_width)

networks[Descriptor('multiply', 1, "linear")] = multiply



for depth in [2, 3]:
	for activation in activations:
		def d(a, b, output_width):

			v = tf.concat([a,b],-1)

			for i in range(depth):
				width = output_width if i == depth-1 else inner_width
				v_new = layer_dense(v, width, activation)

				# This will not do residual on the first layer nor the last layer
				# Therefore residual will only kick in on depth 3+
				if v.shape == v_new.shape:
					v = v + v_new
				else:
					v = v_new

			return v

		networks[Descriptor('dense_residual', depth, activation)] = d



