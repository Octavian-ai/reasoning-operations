

import tensorflow as tf
import numpy as np
from collections import namedtuple

from .datasets import bus_width


Task = namedtuple('Task', ['expected_fn', 'output_width'])

tasks = {
	"concat":			Task(lambda a, b: tf.concat([a,b], -1) * tf.get_variable("bias", [1,1], dtype=tf.float64), bus_width*2),
	"equality":     	Task(lambda a, b: tf.cast(tf.equal(a, b),       tf.float32), bus_width),
	"logical_and":  	Task(lambda a, b: tf.cast(tf.logical_and(tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32), bus_width),
	"logical_or":  		Task(lambda a, b: tf.cast(tf.logical_or( tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32), bus_width),
	"logical_xor":  	Task(lambda a, b: tf.cast(tf.logical_xor(tf.cast(a, tf.bool), tf.cast(b, tf.bool)), tf.float32), bus_width),
	"elementwise_add":  Task(lambda a, b: tf.add(a, b), bus_width),
	"elementwise_mul":  Task(lambda a, b: tf.multiply(a, b), bus_width),
	"reduce_sum":  		Task(lambda a, b: tf.reduce_sum(tf.concat([a, b], -1), -1, keepdims=True), 1),
	"reduce_max":  		Task(lambda a, b: tf.reduce_max(tf.concat([a, b], -1), -1, keepdims=True), 1),
}
