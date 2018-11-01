
import tensorflow as tf
import numpy as np
from collections import namedtuple
import csv

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from .datasets import *
from .tasks import *
from .networks import *


def run_experiment(task, network, gen_dataset, batch_size=32, learning_rate=1e-1, training_steps=20*1000, accuracy_places=4, lr_decay_rate=1.0, model_dir=None):

	dataset = gen_dataset()

	assert len(dataset.train.shape) == 3, f"Train dataset missing dimensions, {dataset.train.shape} {dataset}"
	assert len(dataset.test.shape) == 3, f"Test dataset missing dimensions, {dataset.test.shape} {dataset}"
	assert dataset.train.shape[1] == 2, f"Train dataset not expected shape {dataset.train.shape}"
	assert dataset.test.shape[1]  == 2, f"Test dataset not expected shape {dataset.test.shape}"

	def model_fn(features, labels, mode, params):

		predictions = network(features[:,0,:], features[:,1,:], task.output_width)
		labels = task.expected_fn(features[:,0,:], features[:,1,:])

		lr = tf.train.exponential_decay(
			learning_rate, tf.train.get_global_step(),
			training_steps, lr_decay_rate)

		loss = tf.losses.mean_squared_error(labels, predictions)
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		metrics = {
			"accuracy": tf.metrics.accuracy(
				labels=tf.zeros(tf.shape(labels)), 
				predictions=tf.round(
					(tf.cast(predictions,tf.float64) - tf.cast(labels,tf.float64)) * 
					tf.cast(tf.pow(10.0,accuracy_places), tf.float64)
				)
			)
		}

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

	def gen_input_fn(data):
		def input_fn():
			dataset = tf.data.Dataset.from_tensor_slices(data)
			return dataset.batch(batch_size)
		return input_fn

	estimator = tf.estimator.Estimator(model_fn=model_fn,params={}, model_dir=model_dir)

	training_steps_calc = round(training_steps * dataset.train.shape[0] / batch_size)

	train_spec = tf.estimator.TrainSpec(input_fn=gen_input_fn(dataset.train), max_steps=training_steps_calc)
	eval_spec = tf.estimator.EvalSpec(input_fn=gen_input_fn(dataset.test))

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	# estimator.train(input_fn=gen_input_fn(dataset.train), steps=training_steps_calc)
	evaluation = estimator.evaluate(input_fn=gen_input_fn(dataset.test), steps=round(dataset.test.shape[0] / batch_size))

	evaluation["accuracy_pct"] = str(round(evaluation["accuracy"]*100))+"%"

	return evaluation

def run_all():

	tf.logging.set_verbosity("ERROR")

	header = ["task", "dataset", "network_type", "network_depth", "network_activation", "accuracy", "loss"]

	with tf.gfile.GFile("./output.csv", "w+") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(header)
		print(header)

		for dk, gen_dataset in datasets.items():
			for tk, task in tasks.items():	
				for nk, network in networks.items():		

					result = run_experiment(task, network, gen_dataset)
					row = [
						tk, dk, nk.type, nk.layers, nk.activation, result["accuracy_pct"], result["loss"]
					]
					writer.writerow(row)
					print(row)

def LRRange(mul=5):
	
	for i in range(mul*6, 0, -1):
		lr = pow(0.1, i/mul)
		yield lr

	for i in range(1, 2*mul+1):
		lr = pow(10, i/mul)
		yield lr


def explore_lr():

	tf.logging.set_verbosity("INFO")

	gen_dataset = datasets["one_hot"]
	task = tasks["concat"]
	network = networks[Descriptor('dense', 1, "linear")]

	# for lr in LRRange():
	lr = 0.630957344480193
	result = run_experiment(task, network, gen_dataset, learning_rate=lr, training_steps=1000, lr_decay_rate=0.0001, model_dir="./model/concat_dense")
	print(lr, result["accuracy_pct"])


if __name__ == "__main__":
	# run_all()
	explore_lr()


