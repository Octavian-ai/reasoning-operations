
import tensorflow as tf
import numpy as np
from collections import namedtuple
import csv
from uuid import uuid4
import os.path
import datetime

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from .datasets import *
from .tasks import *
from .networks import *
from .hooks import *

ACCURACY_TARGET = 0.99


def run_experiment(task, network, gen_dataset, batch_size=32, learning_rate=1e-1, training_steps=20*1000, accuracy_places=4, lr_decay_rate=1.0, model_dir=None,eval_every=60):

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
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		metrics = {
			"accuracy": tf.metrics.accuracy(
				labels=tf.zeros(tf.shape(labels)), 
				predictions=tf.round(
					(tf.cast(predictions,tf.float64) - tf.cast(labels,tf.float64)) * 
					tf.cast(tf.pow(10.0,accuracy_places), tf.float64)
				)
			),
			"learning_rate": tf.metrics.mean(lr),
		}

		hooks = [
			EarlyStoppingHook(metrics["accuracy"], ACCURACY_TARGET)
		]

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics, training_hooks=hooks)

	def gen_input_fn(data):
		# def input_fn():
		# 	dataset = tf.data.Dataset.from_tensor_slices(data)
		# 	dataset = dataset.batch(batch_size)
		# 	dataset = dataset.repeat()
		# 	return dataset

		return tf.estimator.inputs.numpy_input_fn(x=data, num_epochs=None, shuffle=True, batch_size=batch_size)
		# return input_fn



	run_config = tf.estimator.RunConfig(save_checkpoints_secs=eval_every)

	estimator = tf.estimator.Estimator(model_fn=model_fn,params={}, model_dir=model_dir, config=run_config)

	train_spec = tf.estimator.TrainSpec(input_fn=gen_input_fn(dataset.train), max_steps=training_steps)
	eval_spec = tf.estimator.EvalSpec(input_fn=gen_input_fn(dataset.test),
		start_delay_secs=5, throttle_secs=eval_every)

	# tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	estimator.train(input_fn=gen_input_fn(dataset.train), steps=training_steps)

	evaluation = estimator.evaluate(input_fn=gen_input_fn(dataset.test), steps=100)

	evaluation["accuracy_pct"] = str(round(evaluation["accuracy"]*100))+"%"

	return evaluation

def run_all():

	tf.logging.set_verbosity("INFO")

	header = ["task", "dataset", "network_type", "network_depth", "network_activation", "accuracy", "loss","datetime"]

	with tf.gfile.GFile("./output.csv", "wa+") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(header)
		print(header)

		for dk, gen_dataset in datasets.items():
			for tk, task in tasks.items():	
				for nk, network in networks.items():	

					setup = [tk, dk, nk.type, nk.layers, nk.activation]	

					print("Finding best result for", setup)

					result = grid_best(task, network, gen_dataset, '_'.join(setup))

					row = setup + [
						result["accuracy_pct"], result["loss"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
					]
					writer.writerow(row)
					writer.flush()
					print(row)

def LRRange(mul=3):
	
	for i in range(mul*6, 0, -1):
		lr = pow(0.1, i/mul)
		yield lr

	for i in range(1, 2*mul+1):
		lr = pow(10, i/mul)
		yield lr


def explore_lr():

	# tf.logging.set_verbosity("INFO")

	gen_dataset = datasets["one_hot"]
	task = tasks["concat"]
	network = networks[Descriptor('dense', 1, "linear")]

	lr = 1.7782794100389232e-05
	gr = 1e6

	# for lr in LRRange():
	result = run_experiment(task, network, gen_dataset, learning_rate=lr, lr_decay_rate=1.0, training_steps=100_000, model_dir=f"./model/concat_dense/{lr}")
	print(lr, result["accuracy_pct"])


def grid_best(task, network, gen_dataset, prefix, use_uuid=False):
	# Important: prefix needs to summarise the run uniquely if !use_uuid!

	results = []

	for lr in LRRange():

		model_dir_parts = ["model", prefix, str(lr)]
		if use_uuid:
			model_dir_parts.append(str(uuid4()))

		model_dir = os.path.join(*model_dir_parts)

		result = run_experiment(task, network, gen_dataset, learning_rate=lr, lr_decay_rate=1.0, training_steps=100_000, model_dir=model_dir)
		print("grid_best", lr, result["accuracy_pct"])
		
		if result["accuracy"] > ACCURACY_TARGET:
			return result

		if len(results) == 0 or result["loss"] < min([i["loss"] for i in results]):
			results.append(result)
		else:
			return min(results, key=lambda i:i["loss"])





if __name__ == "__main__":
	run_all()
	# explore_lr()


