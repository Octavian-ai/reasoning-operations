
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

NetworkDescriptor = namedtuple('NetworkDescriptor', ['type', 'layers', 'activation'])


# Python doesn't do variable scoping as one would imagine grrrr
# this is to capture the scope properly and not overwrite
# values in a bizare way

def add_dense_network(depth, activation, include_residual=False):
	def d(a, b, output_width):
		v = tf.concat([a,b],-1)
		for i in range(depth):
			width = output_width if i == depth-1 else inner_width
			v_new = layer_dense(v, width, activation)

			# This will not do residual on the first layer nor the last layer
			# Therefore residual will only kick in on depth 3+
			if v.shape == v_new.shape and include_residual:
				v = v + v_new
			else:
				v = v_new

		return v

	r = "_residual" if include_residual else ""
	networks[NetworkDescriptor('dense'+r, depth, activation)] = d

# Already run
# for depth in range(1, max_depth+1):
# 	for activation in activations:
# 		add_dense_network(depth, activation)


def multiply(a, b, output_width):
	a = layer_dense(a, inner_width)
	b = layer_dense(b, inner_width)
	c = tf.multiply(a,b)
	return layer_dense(c, output_width)

# Already run
# networks[NetworkDescriptor('multiply', 1, "linear")] = multiply


for depth in [2, 3]:
	for activation in activations:
		add_dense_network(depth, activation, True)



