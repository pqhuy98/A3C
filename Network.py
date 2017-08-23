from Environment import Environment
from Model import ModelVP

from keras import backend as K
import tensorflow as tf
import numpy as np
import time, threading

class Network() :
	"""
		Basic funtions and similar properties of Brain and Agent.
	"""

	def __init__(self, env, early_skipping, use_LSTM, name = "") :
		"""
			@env : The environment contains this network...
			@early_skipping : skip the game for some steps.
		"""
		# ----------------Own environment-----------------------
		self.env = env
		self.early_skipping = early_skipping
		# -----------------Networks and trainable weights-----------
		with tf.variable_scope(str(name)) :
			# Networks symbols
			self.state, self.V, self.P, self.action, self.reward, 	\
			self.get_grad, self.weights, self.model, 				\
			self.total_loss, self.lossV, self.lossP, self.A 		\
			= ModelVP(self.env.shape, self.env.n_action, use_LSTM)
			"""
			self.state : environment state (TF placeholder).
			self.V, self.P : value and policy probs output (TF ops).
			self.action : one hot action vector (TF placeholder).
			self.reward : estimated reward (TF placeholder).
			self.get_grad : gradient calculating (TF op).
			self.weights : trainable weights (Keras variables).
			self.model : Keras model.
			self.total_loss, self.lossV, self.lossP : losses (TF ops).
			self.A : advantage value (TF op).
			"""

			# Session
			self.sess = K.get_session()

		self.reset_game(self.early_skipping)
		# threading.Event() -- used to stop proccess immediately.
		self.event = threading.Event()

	def act(self) :
		V, P = self.model.predict(np.array([self.env.state]))
		V, P = V[0], P[0]
		action = np.random.choice(len(P), p=P)
		o, r, done, info = self.env.make_action(action)
		return (o, r, done, info), action, (V, P)

	def reset_game(self, early_skipping = None) :
		if (early_skipping is None) :
			self.env.reset(self.early_skipping)
		else :
			self.env.reset(early_skipping)
		self.model.reset_states()

	def save(self, path="best") :
		self.model.save_weights(path+"_weights.h5")

	def load(self, path="best") :
		self.model.load_weights(path+"_weights.h5")

	def stop(self) : # Stop the game immediately.
		self.event.clear()

#---For-debugging-----------------------------------------------------------
	def get_loss(self, s, a, r) :
		feed_dict = {self.state: s, self.action: a, self.reward: r}
		return self.sess.run(self.total_loss, feed_dict=feed_dict)

	def get_lossV(self, s, a, r) :
		feed_dict = {self.state: s, self.action: a, self.reward: r}
		return self.sess.run(self.lossV, feed_dict=feed_dict)

	def get_lossP(self, s, a, r) :
		feed_dict = {self.state: s, self.action: a, self.reward: r}
		return self.sess.run(self.lossP, feed_dict=feed_dict)

	def get_V(self, s) : # This one is used.
		feed_dict = {self.state: s}
		return self.sess.run(self.V, feed_dict=feed_dict)[0, 0]

	def get_P(self, s) :
		feed_dict = {self.state: s}
		return self.sess.run(self.P, feed_dict=feed_dict)

	def get_A(self, s, r) :
		feed_dict = {self.state: s, self.reward: r}
		return self.sess.run(self.A, feed_dict=feed_dict)

#---Testing-----------------------------------------------------------------	
if __name__ == "__main__" :
	e = Environment("Pong-v0", 4, "default")
	A = Network(env = e)

	while True :
		(o, r, done, info), a, _ = A.act()
		A.env.render()
		if (done) :
			A.reset_game()