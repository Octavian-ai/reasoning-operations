
import tensorflow as tf

class EarlyStoppingHook(tf.train.SessionRunHook):

	def __init__(self, metric, target=0.99999999999, check_every=100, start_time=None, max_secs=None):
		self.metric = metric
		self.target = target
		self.counter = 0
		self.check_every = check_every
		self.max_secs = max_secs
		self.start_time = start_time
		
	def before_run(self, run_context):
		self.counter += 1
		self.should_check = (self.counter % self.check_every) == 0

		if self.should_check:
			return tf.train.SessionRunArgs([self.metric])

	def after_run(self, run_context, run_values):
		if self.should_check and run_values.results is not None:
			t = run_values.results[0][1]
			if t > self.target:
				tf.logging.info(f"Early stopping as exceeded target {t} > {self.target}")
				run_context.request_stop()

		if self.max_secs is not None and self.start_time is not None and (time.time() - self.start_time) > self.max_secs:
			tf.logging.info(f"EarlyStopping as time run out {time.time() - self.start_time} > {self.max_secs}")
			run_context.request_stop()
		