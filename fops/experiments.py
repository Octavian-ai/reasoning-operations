
import tensorflow as tf
import numpy as np
from collections import namedtuple

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from .datasets import *
from .tasks import *
from .networks import *


def run_experiment(task, network, gen_dataset, batch_size=32, learning_rate=1e-2, training_steps=100):

	dataset = gen_dataset()

	assert len(dataset.test.shape) == 3, f"Test dataset missing dimensions, {dataset.test.shape} {dataset}"
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
	evaluation = estimator.evaluate(input_fn=lambda:dataset.test, steps=dataset.test.shape[0])

	return evaluation["accuracy"]

def run_all():

	tf.logging.set_verbosity("ERROR")

	header = ["task", "dataset", "network_type", "network_depth", "network_activation", "accuracy", "loss"]

	with tf.gfile.GFile("./output.csv", "w+") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(header)

		for tk, task in tasks.items():
			for nk, network in networks.items():
				for dk, gen_dataset in datasets.items():
					result = run_experiment(task, network, gen_dataset)
					row = [
						tk, dk, nk.type, nk.layers, nk.activation, result["accuracy"], result["loss"]
					]
					writer.writerow(row)
					print(row)


if __name__ == "__main__":
	run_all()


