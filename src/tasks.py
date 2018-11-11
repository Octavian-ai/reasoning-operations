

import tensorflow as tf
import numpy as np
from collections import namedtuple

from .datasets import bus_width


Task = namedtuple('Task', ['expected_fn', 'output_width'])

def one_hot_sum(a,b):
	a_int = tf.argmax(a, -1)
	b_int = tf.argmax(b, -1)

	c_int = a_int + b_int

	return tf.one_hot(c_int, bus_width)

def batch_dot(a,b):
	v = tf.multiply(a,b)
	return tf.reduce_sum(v, -1, keepdims=True)

def almost_equal(a,b):
	delta = tf.abs(a - b)
	return tf.cast(tf.less_equal(delta, 0.001), tf.float32)

tasks = {
	"reduce_sum":  		Task(lambda a, b: tf.reduce_sum(tf.concat([a, b], -1), -1, keepdims=True), 1),
	"reduce_max":  		Task(lambda a, b: tf.reduce_max(tf.concat([a, b], -1), -1, keepdims=True), 1),
	"concat":			Task(lambda a, b: tf.concat([a,b], -1), bus_width*2),
	"dot":				Task(batch_dot, 1),
	"elementwise_mul":  Task(lambda a, b: tf.multiply(a, b), bus_width),
	"one_hot_sum":		Task(one_hot_sum, bus_width),
	"elementwise_add":  Task(lambda a, b: tf.add(a, b), bus_width),
	"equality":     	Task(lambda a, b: tf.cast(tf.equal(a, b),       tf.float32), bus_width),
	"almost_equal":		Task(almost_equal, bus_width),
	"logical_and":  	Task(lambda a, b: tf.cast(tf.logical_and(tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32), bus_width),
	"logical_or":  		Task(lambda a, b: tf.cast(tf.logical_or( tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32), bus_width),
	"logical_xor":  	Task(lambda a, b: tf.cast(tf.logical_xor(tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32), bus_width),
}
