
import tensorflow as tf
import numpy as np
from collections import namedtuple

from .datasets import *
from .tasks import *
from .networks import *


def run_experiment(task, network, gen_dataset, batch_size=32, learning_rate=1e-3, training_steps=10*1000):

	dataset = gen_dataset()

	assert dataset.train.shape[1] == 2, f"Train dataset not expected shape {dataset.train.shape}"
	assert dataset.test.shape[1]  == 2, f"Test dataset not expected shape {dataset.test.shape}"

	def model_fn(features, labels, mode, params):

		predictions = network(features[:,0,:], features[:,1,:], task.output_width)
		labels = task.expected_fn(features[:,0,:], features[:,1,:])

		loss = tf.losses.mean_squared_error(labels, predictions)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		metrics = {
			"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
		}

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


	estimator = tf.estimator.Estimator(model_fn=model_fn,params={})
	estimator.train(input_fn=lambda:dataset.train, steps=training_steps)
	evaluation = estimator.evaluate(input_fn=lambda:dataset.test)

	return evaluation["accuracy"]




for tk, task in tasks.items():
	for nk, network in networks.items():
		for dk, gen_dataset in datasets.items():
			result = run_experiment(task, network, gen_dataset)
			print(tk, nk, dk, result)


