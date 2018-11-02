
import tensorflow as tf

from .activations import *

def layer_selu(tensor, width, dropout=0.0, name=None):

	if name is None:
		name_dense = None
		name_drop = None
	else:
		name_dense = name + "_dense"
		name_drop = name + "_drop"

	r = tf.layers.dense(tensor, width, 
		activation=tf.nn.selu,
		kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0), 
		name=name_dense)

	if dropout > 0.0:
		r = tf.contrib.nn.alpha_dropout(r, dropout, name=name_drop)

	return r

def layer_dense(tensor, width:int, activation_str:str="linear", dropout:float=0.0, name:str=None):

	if activation_str == "selu":
		return layer_selu(tensor, width, dropout, name)
	else:
		v = tf.layers.dense(tensor, width, activation=ACTIVATION_FNS[activation_str], name=name)

		if dropout > 0:
			v = tf.nn.dropout(v, 1.0-dropout)

		return v
