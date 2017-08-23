from scipy.misc import imresize, imsave
import numpy as np
import gym, cv2
import time

class Environment() :
	"""
		Extended from gym environments. Can be modified to many other kinds.
		Support time channel (multiple frames in a single observation).
		------------Main menthods------------
		@make_action(action) : do action and receive game status (obs, reward, done, info).
		@reset() : start a new fresh environment.
		@shape : observations shape.
		@n_action : number of possible actions.
		@state : current state.
		@score : current score.
	"""

	def __init__(self, env_name, nb_frames, preprocess = None) :
		"""
		@env_name :
			Gym environment name, eg. "Pong-v0", "Breakout-v0",...
		@preprocess : 
			Function to normalize raw game states (eg. resize, RGB->gray...).
			See @default_preprocess belows.
		@nb_frames :
			Each step consists of # raw frames stacked. Time channel is the last dimension.
			Example :
			Raw RGB game image : (210, 160, 3),
			Resize and convert to gray : (80, 80), 
			@nb_frames = 5
			=> New state is (80, 80, 5).
		"""
		self.preprocess = preprocess
		if (preprocess == "default") :
			self.preprocess = self.default_preprocess

		self.nb_frames = nb_frames

		self.env = gym.make(env_name)
		self.state = None
		self.score = None
		self.done = None
		self.info = None
		self.reset()

		self.shape = self.state.shape
		self.n_action = self.env.action_space.n

	def make_action(self, action) :
		reward = 0
		res = np.zeros(self.state.shape)
		for i in range(self.nb_frames) :
			pre_info = self.info
			o, r, self.done, self.info = self.env.step(action)
			if (self.preprocess is not None) :
				o = self.preprocess(o)
			res[..., i] = o
			r+= self.info["ale.lives"] - pre_info["ale.lives"]
			reward+= r

		self.state = res
		self.score+= reward
		return res, reward, self.done, self.info

	def reset(self, early_skipping = 0) :
		"""
		Reset the environment. Also skip $early_skipping steps.
		"""
		o = self.env.reset()
		if (self.preprocess is not None) :
			o = self.preprocess(o)

		# Dirty hack
		self.state = np.repeat(np.expand_dims(o, -1), self.nb_frames, axis=-1)
		self.info = {"ale.lives":5}
		self.score = 0
		self.done = False
		for i in range(self.nb_frames*(early_skipping-1)) :
			o, r, self.done, info = self.env.step(self.random_action())
			self.score+= r
		if (early_skipping) :
			self.make_action(self.random_action())
		if (self.done) :
			raise Exception("Too muh early skipping : %i."%early_skipping)


	def default_preprocess(self, rgb) :
		new_size = 64
		# r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		# gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		# gray = imresize(gray, (new_size,new_size))
		# return gray/127.5-1
		x = rgb[32:, 8:-8, :]
		x = x[:,:,0] + x[:,:,1] + x[:,:,2]
		x = imresize(x, (new_size,new_size))
		x[x>0] = 1
		return x

	def random_action(self) :
		return self.env.action_space.sample()

	def render(self) :
		self.env.render()

#---Testing-----------------------------------------------------------------
if __name__ == "__main__" :
	e = Environment("Breakout-v0", 10, preprocess="default")
	while True :
		o, r, done, info = e.make_action(e.random_action())
		for i in range(10) :
			imsave("%i.png"%i, o[..., i])
		print o.shape, r, done
		if (done) :
			e.reset()
		# time.sleep(0.01)