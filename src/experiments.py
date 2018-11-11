
import tensorflow as tf
import numpy as np
from collections import namedtuple
import csv
from uuid import uuid4
import os.path
import datetime
import itertools

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

from .datasets import *
from .tasks import *
from .networks import *
from .hooks import *

ACCURACY_TARGET = 0.99


def run_experiment(task, network, gen_dataset, training_steps, 
	prefix_parts,
	batch_size=32, learning_rate=1e-1, accuracy_places=0.5, lr_decay_rate=1.0, eval_every=60,
	predict=False):

	tf.logging.info("Generating dataset")

	dataset = gen_dataset()

	model_dir = os.path.join("model", *prefix_parts)

	assert len(dataset.train.shape) == 3, f"Train dataset missing dimensions, {dataset.train.shape} {dataset}"
	assert len(dataset.test.shape) == 3, f"Test dataset missing dimensions, {dataset.test.shape} {dataset}"
	assert dataset.train.shape[1] == 2, f"Train dataset not expected shape {dataset.train.shape}"
	assert dataset.test.shape[1]  == 2, f"Test dataset not expected shape {dataset.test.shape}"

	def model_fn(features, labels, mode, params):

		with tf.name_scope("network"):
			predictions = network(features[:,0,:], features[:,1,:], task.output_width)

		with tf.name_scope("task"):
			labels = task.expected_fn(features[:,0,:], features[:,1,:])

		with tf.name_scope("loss_training"):
			lr = tf.train.exponential_decay(
				learning_rate, tf.train.get_global_step(),
				training_steps, lr_decay_rate)

			loss = tf.losses.mean_squared_error(labels, predictions)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		delta_vec = tf.round(
			(tf.cast(predictions,tf.float64) - tf.cast(labels,tf.float64)) * 
			tf.cast(tf.pow(10.0,accuracy_places), tf.float64))

		metrics = {
			"accuracy": tf.metrics.accuracy(
				labels=tf.zeros(tf.shape(labels)), 
				predictions=delta_vec,
			),
			"lr": tf.metrics.mean(lr),
		}

		hooks = [
			EarlyStoppingHook(metrics["accuracy"], ACCURACY_TARGET)
		]

		predictions = {
			"features": features,
			"prediction": predictions,
			"label": labels,
			"delta_vec": delta_vec,
		}

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics, predictions=predictions, training_hooks=hooks)

	def gen_input_fn(data, repeat=True):
		# def input_fn():
		# 	dataset = tf.data.Dataset.from_tensor_slices(data)
		# 	dataset = dataset.batch(batch_size)
		# 	dataset = dataset.repeat()
		# 	return dataset

		num_epochs = None if repeat else 1

		return tf.estimator.inputs.numpy_input_fn(x=data, num_epochs=num_epochs, shuffle=True, batch_size=batch_size)
		# return input_fn



	run_config = tf.estimator.RunConfig(save_checkpoints_secs=eval_every)

	estimator = tf.estimator.Estimator(model_fn=model_fn,params={}, model_dir=model_dir, config=run_config)

	# train_spec = tf.estimator.TrainSpec(input_fn=gen_input_fn(dataset.train), max_steps=training_steps)
	# eval_spec = tf.estimator.EvalSpec(input_fn=gen_input_fn(dataset.test, False),
	# 	start_delay_secs=5, throttle_secs=eval_every)

	# tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	estimator.train(input_fn=gen_input_fn(dataset.train), max_steps=training_steps)

	evaluation = estimator.evaluate(input_fn=gen_input_fn(dataset.test, False))

	if predict:
		results = estimator.predict(input_fn=gen_input_fn(dataset.test, False))
		for idx, i in itertools.islice(enumerate(results), 3):
			print(i)
	
	evaluation["accuracy_pct"] = str(round(evaluation["accuracy"]*100))+"%"

	return evaluation

def run_all(training_steps=30_000):

	tf.logging.set_verbosity("ERROR")

	header = ["task", "dataset", "network_type", "network_depth", "network_activation",
		"accuracy_pct", "accuracy", "loss","lr", "global_step", "datetime"]

	with tf.gfile.GFile("./output.csv", "a+") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(header)
		print(header)

		results = []

		for nk, network in networks.items():
			for dk, gen_dataset in datasets.items():
				for tk, task in tasks.items():
					try:
						setup = [tk, dk, nk.type, str(nk.layers), nk.activation]	

						print("Finding best result for", setup)

						if len(results) > 0:
							lslr = results[-1]["lr"]
						else:
							lslr = None

						result = grid_then_long(task, network, gen_dataset, setup, training_steps=training_steps, last_successful_lr=lslr)

						row = setup + [
							result["accuracy_pct"], result["accuracy"], result["loss"], 
							result["lr"], result["global_step"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
						]
						results.append(result)
						writer.writerow(row)
						csvfile.flush()
						print(row)
					except:
						pass

def LRRange(mul=3):

	# yield 1.0000000000000003e-05
	# yield 2.1544346900318827e-05
	yield 0.000001
	yield 0.00001
	yield 0.0001
	# yield 0.00021544346900318848
	# yield 0.00046415888336127784
	yield 0.001
	# yield 0.0021544346900318847
	# yield 0.004641588833612778
	yield 0.01
	# yield 0.021544346900318836
	# yield 0.0464158883361278
	yield 0.1
	# yield 0.2154434690031884
	# yield 0.4641588833612779
	yield 1.0
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


def grid_then_long(task, network, gen_dataset, prefix_parts, training_steps=10_000, last_successful_lr=None, **kwargs):

	# tf.logging.set_verbosity("INFO")

	result = grid_best(task, network, gen_dataset, 
		prefix_parts, 
		training_steps=1000, last_successful_lr=last_successful_lr, 
		**kwargs)

	if result["accuracy"] > ACCURACY_TARGET:
		return result

	prefix_parts = prefix_parts + [str(result["lr"]), "10_000"]

	result2 = run_experiment(task, network, gen_dataset, 
		prefix_parts=prefix_parts,
		training_steps=training_steps, learning_rate=result["lr"], 
		**kwargs)

	if result2["accuracy"] > result["accuracy"]:
		return result2
	else:
		return result


def grid_best(task, network, gen_dataset, prefix_parts, use_uuid=False, improvement_error_threshold=0.7, training_steps=10_000, last_successful_lr=None, **kwargs):

	# Important: prefix needs to summarise the run uniquely if !use_uuid!

	results = []

	def lslr():
		if last_successful_lr is not None:
			yield last_successful_lr

	decreases = 0

	for lr in itertools.chain(lslr(), LRRange()):

		prefix_parts = [*prefix_parts, str(lr), str(training_steps)]
		if use_uuid:
			prefix_parts.append(str(uuid4()))

		result = run_experiment(task, network, gen_dataset, 
			prefix_parts=prefix_parts,
			learning_rate=lr, lr_decay_rate=1.0, 
			training_steps=training_steps,
			**kwargs)

		result["lr"] = lr # Remove rounding corruption etc
		print("grid_best", lr, result["accuracy_pct"])
		
		if result["accuracy"] >= ACCURACY_TARGET:
			return result

		results.append(result)

		if len(results) >= 2:
			if results[-1]["accuracy"] < results[-2]["accuracy"] * improvement_error_threshold:
				logger.info("Accuracy decreased from last run")
				decreases += 1

		if decreases >= 2:
			logger.info("Stopping LR search as acc decreased twice")
			break

	return min(results, key=lambda i:i["loss"])


def run_just_one():

	# tf.logging.set_verbosity("INFO")

	task = tasks["logical_xor"]
	gen_dataset = datasets["many_hot"]
	network = networks[NetworkDescriptor('dense', 2, 'selu')]

	evaluation = run_experiment(task, network, gen_dataset, 
		training_steps=30_000,
		learning_rate=0.001,
		prefix_parts=["run_just", str(uuid4())],
		predict=True,
		)

	print(evaluation)

if __name__ == "__main__":
	run_all()


