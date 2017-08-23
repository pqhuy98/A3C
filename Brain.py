from keras import backend as K
from Network import Network
import tensorflow as tf
import numpy as np
import threading
import gym, cv2

class Brain(Network) :
	"""
	The central network gets gradients from Agents and send back the lastest version.
	-------------Main menthods-------------
	@train(n_step) :
		Play the game for some steps.
		Returns game status, gradients (in numpy arrays) and total loss.
	"""



	def __init__(self, env, early_skipping, use_LSTM, optimizer="default", name = "") :
		"""
			@env_name : gym environment name, eg. "Pong-v0", "Breakout-v0"...
			@optimizer : a tensorflow optimizer used to do gradients descent.
		"""
		Network.__init__(self, env, early_skipping, use_LSTM, name)
		with tf.variable_scope(name) :
			# Update gradients operation.
			self.gradients = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in self.weights]
			self.grads_and_vars = zip(self.gradients, self.weights)

			if (optimizer == "default") :
				self.optimizer = self.default_optimizer()
			else :
				self.optimizer = optimizer

			self.update_grads = [
				self.optimizer.apply_gradients(self.grads_and_vars),
				self.global_step.assign(self.global_step+1)
			]


			self.sess = K.get_session()

	def default_optimizer(self, lr=1e-3, lr_decay=0.96, step=10000, rmsprop_decay=0.99) :
		self.global_step = tf.Variable(0, trainable=False)
		self.learning_rate = tf.train.exponential_decay(lr, self.global_step, step, lr_decay, staircase=False)
		return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=rmsprop_decay)

	def get_weights(self) :
		weights = self.sess.run(self.weights)
		return weights

	def apply_gradients(self, gradients) :
		feed_dict = {}

		if (len(self.gradients) != len(gradients)) :
			l1 = len(self.gradients)
			l2 = len(gradients)
			s = "Apply gradients : length of gradients lists do not match : %i vs %i !"%(l1,l2)
			raise Exception(s)

		for x, y in zip(self.gradients, gradients) :
			if (x.get_shape() != y.shape) :
				s1 = x.get_shape()
				s2 = y.shape
				raise Exception("Gradients shape do not match : %s vs %s"%(s1,s2))
			else :
				feed_dict.update({x:y})

		self.sess.run(self.update_grads, feed_dict=feed_dict)

#---Testing-----------------------------------------------------------------
if __name__ == "__main__" :
	from Environment import Environment
	e = Environment("Pong-v0", 4, "default")
	network = Brain(e, "default")

