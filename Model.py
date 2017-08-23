#------------Allow GPU memory growth---------------------
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
#--------------------------------------------------------

from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GRU, LSTM, Lambda
from keras.layers import Activation, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Reshape
from keras.models import Model
from keras import backend as K

import numpy as np

def ModelVP(obs_shape, n_action, use_LSTM, entropy_weight=1e-2, model_only=False) :
	"""
	Build the neural net and returns its informations.
	Input : (None, H, W, C)
		Batch dimension is equivalent to time dimentsion.
	Shapes :
		V(None, 1), P(None, n_action), action(None, n_action), reward(None, 1), A(None, 1)
	"""

	state = Input(shape=obs_shape)
	x = state # (None, H, W ,C)

	if (use_LSTM) : # Build CNN with LSTM through time.
		x = Lambda(lambda x: K.expand_dims(x, 0))(x) # (1, None, H, W, C)
		x = TimeDistributed(Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu"))(x)
		x = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu"))(x)
		x = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu"))(x)
		x = TimeDistributed(Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu"))(x)
		# (1, None, H/8, W/8, 32)

		x = TimeDistributed(Flatten())(x) # (1, None, H/8 * W/8 * 32)

		x = LSTM(128, stateful=True, return_sequences=True)(x)	# (1, None, 128)

		V = TimeDistributed(Dense(1)) (x)						# (1, None, 1)
		V = Lambda(lambda x: K.reshape(x, (-1, 1))) (V) 		# (None, 1)

		P = TimeDistributed(Dense(n_action, activation="softmax")) (x)	# (1, None, n_action)
		P = Lambda(lambda x: K.reshape(x, (-1, n_action))) (P) 				# (None, n_action)

	else :
		x = Conv2D(16, (5, 5), strides=(2, 2), padding="same", activation="relu")(x)
		x = Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu")(x)
		x = Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu")(x)
		x = Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation="relu")(x)
		x = Dense(256, activation="relu")(Flatten()(x))

		V = Dense(1)(x)
		P = Dense(n_action, activation="softmax")(x)

	model = Model(state, [V, P])
	model._make_predict_function()

	action = tf.placeholder(tf.float32, shape=(None,n_action)) # One hot vector of chosen action.
	reward = tf.placeholder(tf.float32, shape=(None,1)) # Estimated reward.

	A = reward-V	#Advantage value

	# Policy probability and its log
	prob = tf.reduce_sum(P*action, axis = 1, keep_dims=True)
	log_prob = tf.log(prob+1e-8)

	# Entropy to encourage exploration.
	lossE = P*tf.log(P+1e-8)

	# Policy loss
	lossP = -log_prob*tf.stop_gradient(A)

	# Value loss
	lossV = tf.square(A) # A = reward - V, how coincident !

	total_loss = tf.reduce_mean(lossV + lossP + entropy_weight*lossE)

	weights = model.trainable_weights
	get_grad = K.gradients(total_loss, weights)

	if model_only :
		return model
	else :
		return 	state, V, P, action, reward, \
				get_grad, weights, model,	 \
				total_loss, lossV, lossP, A

#---Testing-----------------------------------------------------------------
if __name__ == "__main__" :
	x = ModelVP((32, 32, 4), 6, True, model_only=True)
	x.summary()