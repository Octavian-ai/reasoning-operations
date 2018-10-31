

import tensorflow as tf
import numpy as np
from collections import namedtuple

from .datasets import bus_width


Task = namedtuple('Task', ['expected_fn', 'output_width'])

tasks = {
	"equality":     	Task(lambda a, b: tf.equal(a, b), bus_width),
	"logical_and":  	Task(lambda a, b: tf.logical_and(a, b), bus_width),
	"logical_or":   	Task(lambda a, b: tf.logical_or(a, b), bus_width),
	"logical_xor":  	Task(lambda a, b: tf.logical_xor(a, b), bus_width),
	"elementwise_add":  Task(lambda a, b: tf.add(a, b), bus_width),
	"elementwise_mul":  Task(lambda a, b: tf.multiply(a, b), bus_width),
}
