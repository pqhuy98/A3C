from keras import backend as K
from Network import Network
import tensorflow as tf
import numpy as np
import threading
import gym

class Agent(Network) :
	"""
	The agent acts inside its own environment. It sends gradient back to Brain.
	-------------Main menthods-------------
	@train(n_step) :
		Play the game for some steps.
		Returns current game status, gradients (in numpy arrays) and total loss.
	@set_weights(new_weights) :
		Update the network with new weights (in numpy arrays).
	"""

	def __init__(self, env, early_skipping, use_LSTM, gamma, name = "") :
		"""
		@env : an Environment object.
		@early_skipping : skip the game for some steps.
		@gamma : future discount.
		"""
		Network.__init__(self, env, early_skipping, use_LSTM, name)
		self.gamma = gamma

		with tf.variable_scope(str(name)) :
			# Recurrent (LSTM/GRU) states, stored for re-calculating gradients.
			self.lstm_states = []
			for l in self.model.layers :
				if (hasattr(l, 'states')) :
					self.lstm_states += l.states

			# array weight operation.
			self.new_weights = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in self.weights]
			self.assign_weights = []
			for i in range(len(self.weights)) :
				op = self.weights[i].assign(self.new_weights[i])
				self.assign_weights.append(op)

			self.sess = K.get_session()

		# threading.Event() -- used to stop proccess immediately.
		self.event = threading.Event()


	def train(self, n_step) :
		"""
		Play the game for some steps.
		Returns game status, gradients (in numpy arrays) and total loss.
		@n_step : number of step to play.
		"""
		self.event.set()
		game_status, (s, a, r, start_lstm_states)  = self.get_new_experience_batch(n_step)
		# @game_status : (total_reward, n_step, done)

		loss, grads = self.get_gradients(s, a, r, start_lstm_states)

		self.event.clear()
		return game_status, loss, grads

	def set_weights(self, new_weights) :
		feed_dict = {}

		if (len(self.new_weights) != len(new_weights)) :
			l1 = len(self.new_weights)
			l2 = len(new_weights)
			s = "Update new weight: length of weights lists do not match : %i vs %i !"%(l1,l2)
			raise Exception(s)

		for x, y in zip(self.new_weights, new_weights) :
			if (x.get_shape() != y.shape) :
				s1 = x.get_shape()
				s2 = y.shape
				raise Exception("Weights shape do not match : %s vs %s"%(s1,s2))
			else :
				feed_dict.update({x:y})

		self.sess.run(self.assign_weights, feed_dict=feed_dict)

	def get_gradients(self, s, a, r, start_lstm_states) :
		# Save current LSTM states.
		current_lstm_states = [K.eval(x) for x in self.lstm_states]

		# Set LSTM states to starting values.
		for (x, y) in zip(self.lstm_states, start_lstm_states) :
			K.set_value(x, y)

		# Calculate gradients
		feed_dict = {self.state: s, self.action: a, self.reward: r}
		loss, grads = self.sess.run([self.total_loss, self.get_grad], feed_dict=feed_dict)

		# Set LSTM states back to current values.
		for (x, y) in zip(self.lstm_states, current_lstm_states) :
			K.set_value(x, y)

		return loss, grads

	def get_new_experience_batch(self, k) :
		"""
		Continue game for at most k steps and return (states, actions, rewards) batch.
		"""
		obs = self.env.state
		total_reward, done = 0, False
		actions_list, states_list, rewards_list = [], [], []
		
		# Save initital LSTM states to calculate gradients later.
		start_lstm_states = [K.eval(x) for x in self.lstm_states]

		for _ in range(k) :
			if (not self.event.is_set()) or done :
				break
			# Follow the policy and generating new batch.
			states_list.append(obs)
			(obs, reward, done, info), action, _ = self.act()
			actions_list.append(action)
			rewards_list.append([reward])
			total_reward+= reward

		# The game went for n_step steps.
		n_step = min(k, len(actions_list))

		# Calculate states batch
		s_batch = np.array(states_list).astype("float32")

		# Calculate actions batch
		a_batch = np.zeros((n_step, self.env.n_action)).astype("float32")
		for i, x in enumerate(actions_list) :
			a_batch[i,x] = 1

		# Calculate rewards batch
		r_batch = np.array(rewards_list).astype("float32")

		if not done :
			r_batch[-1, 0]+= self.gamma*self.get_V([obs])
		else :
			self.reset_game()

		for t in reversed(range(len(rewards_list)-1)) :
			# R(t) = r(t) + R(t+1)*gamma
			r_batch[t,0] = r_batch[t,0] + r_batch[t+1, 0] * self.gamma

		return 	(total_reward, n_step, done), (s_batch, a_batch, r_batch, start_lstm_states)

#---Testing-----------------------------------------------------------------	
if __name__ == "__main__" :
	from Environment import Environment
	from Brain import Brain
	import time

	e1 = Environment("Pong-v0", 4, "default")
	# e2 = Environment("Pong-v0", 4, "default")
	agent = Agent(e1, 15, 0.99)
	# brain = Brain(e2, 15, "default")

	while True :
		# global_weights = brain.get_weights()
		# agent.set_weights(global_weights)

		_, loss, grads = agent.train(10)
		# brain.apply_gradients(grads)
		print loss

		e1.render()
		time.sleep(0.01)
