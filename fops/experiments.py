
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


def run_experiment(task, network, gen_dataset, training_steps, batch_size=32, learning_rate=1e-1, accuracy_places=2, lr_decay_rate=1.0, model_dir=None,eval_every=60):

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

def run_all(training_steps=10_000):

	tf.logging.set_verbosity("ERROR")

	header = ["task", "dataset", "network_type", "network_depth", "network_activation", "accuracy_pct", "accuracy", "loss","lr","datetime"]

	with tf.gfile.GFile("./output.csv", "a+") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(header)
		print(header)

		for dk, gen_dataset in datasets.items():
			for tk, task in tasks.items():	
				for nk, network in networks.items():	

					setup = [tk, dk, nk.type, str(nk.layers), nk.activation]	

					print("Finding best result for", setup)

					result = grid_then_long(task, network, gen_dataset, setup, training_steps=training_steps)

					row = setup + [
						result["accuracy_pct"], result["accuracy"], result["loss"], result["lr"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
					]
					writer.writerow(row)
					csvfile.flush()
					print(row)

def LRRange(mul=3):

	# yield 1.0000000000000003e-05
	# yield 2.1544346900318827e-05
	yield 1.7782794100389232e-05
	yield 4.641588833612783e-05
	yield 0.00010000000000000002
	# yield 0.00021544346900318848
	# yield 0.00046415888336127784
	yield 0.0010000000000000002
	# yield 0.0021544346900318847
	# yield 0.004641588833612778
	yield 0.010000000000000002
	# yield 0.021544346900318836
	# yield 0.0464158883361278
	yield 0.1
	# yield 0.2154434690031884
	# yield 0.4641588833612779
	yield 2.154434690031884
	# yield 4.641588833612778
	yield 10.0
	# yield 21.544346900318832
	# yield 46.4158883361278
	yield 100.0
	
	# for i in range(mul*5, 0, -1):
	# 	lr = pow(0.1, i/mul)
	# 	yield lr

	# for i in range(1, 2*mul+1):
	# 	lr = pow(10, i/mul)
	# 	yield lr


def grid_then_long(task, network, gen_dataset, prefix_parts, training_steps=10_000):

	# tf.logging.set_verbosity("INFO")

	result = grid_best(task, network, gen_dataset, prefix_parts, training_steps=100)
	print(result)
	if result["accuracy"] > ACCURACY_TARGET:
		return result
		
	model_dir = os.path.join("model", *prefix_parts, result["lr"], "10_000")
	result = run_experiment(task, network, gen_dataset, training_steps=training_steps, learning_rate=result["lr"], model_dir=model_dir)
	print(result)

	return result


def grid_best(task, network, gen_dataset, prefix_parts, use_uuid=False, improvement_error_threshold=0.1, training_steps=10_000):

	# Important: prefix needs to summarise the run uniquely if !use_uuid!

	results = []

	for lr in LRRange():

		model_dir_parts = ["model", *prefix_parts, str(lr), str(training_steps)]
		if use_uuid:
			model_dir_parts.append(str(uuid4()))

		model_dir = os.path.join(*model_dir_parts)

		result = run_experiment(task, network, gen_dataset, learning_rate=lr, lr_decay_rate=1.0, training_steps=training_steps, model_dir=model_dir)
		result["lr"] = lr
		print("grid_best", lr, result["accuracy_pct"])
		
		if result["accuracy"] > ACCURACY_TARGET:
			return result

		if len(results) == 0 or result["accuracy"] >= max([i["accuracy"] for i in results]) - improvement_error_threshold:
			results.append(result)
		else:
			break

	return max(results, key=lambda i:i["accuracy"])



if __name__ == "__main__":
	run_all(training_steps=10_000)
	# explore_lr()


