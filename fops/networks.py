
import tensorflow as tf
import numpy as np
from collections import namedtuple

from .layers import *
from .datasets import bus_width

max_depth = 2

activations = ["linear", "tanh", "relu", "selu", 'abs', 'tanh_abs', "sigmoid", "softmax"]

networks = {
}

Descriptor = namedtuple('Descriptor', ['type', 'layers', 'activation'])

def concat(a, b, output_width):
	v = tf.concat([a,b], -1)
	if v.shape[-1] != output_width:
		v = tf.layers.dense(v, output_width)
	return v

networks[Descriptor('concat', 1, "linear")] = concat

for depth in range(1, max_depth):
	for activation in activations:
		def d(a, b, output_width):
			v = tf.concat([a,b],-1)
			for i in range(depth):
				v = layer_dense(v, bus_width, activation)

			if v.shape[-1] != output_width:
				v = layer_dense(v, output_width, "linear")
			return v

		networks[Descriptor('dense', depth, activation)] = d


for depth in range(1, max_depth):
	for activation in activations:
		def d(a, b, output_width):
			v = tf.concat([a,b],-1)
			for i in range(depth):
				v_new = layer_dense(v, bus_width, activation)
				if v.shape == v_new.shape:
					v = v + v_new

			if v.shape[-1] != output_width:
				v = layer_dense(v, output_width, "linear")
			return v

		networks[Descriptor('dense_residual', depth, activation)] = d



