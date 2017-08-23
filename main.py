# Disable CUDA, so don't use GPU.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#--------------------------------------------------------------------------
from Environment import Environment
from Brain import *
from Agent import *

from mpi4py import MPI
import threading
import time

from keras.utils.generic_utils import Progbar
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# Important constants-------------------------------------------------------
nb_frames = 4 # Each step has # frames.
early_skipping = 0 # Skip the first # steps every game. Why waste our time ?
save_period = 200 # Save the network every # games.

# (LOAD_PATH == None) means don't load old model.
LOAD_PATH = None
# LOAD_PATH = "save_breakout/best"
SAVE_PATH = "save_breakout/best"

game_name = "Breakout-v0"
# game_name = "Pong-v0"
play = True
LSTM = False
#---------------------------------------------------------------------------

# MPI initialization.
comm = MPI.COMM_WORLD
n_node = comm.Get_size()
my_id = comm.Get_rank()
my_name = MPI.Get_processor_name()
MASTER = 0
others = range(n_node)
others.remove(my_id)

#---Fast way to send a list of numpy arrays-------------------------------------
def _sendFLOAT(data, dest, tag) :
	for a in data :
		comm.Send([a, MPI.FLOAT], dest=dest, tag=tag)
def sendFLOAT(data, dest, tag) :
	threading.Thread(target=_sendFLOAT, args = (data, dest, tag)).start()

def recvFLOAT(data, src, tag) :
	for a in data :
		comm.Recv([a, MPI.FLOAT], source=src, tag=tag)
#--------------------------------------------------------------------------

def learner(idx, agent) :
	"""
	Thread learner. Run on non-MASTER nodes.
	Play games, calculate gradients, send them to MASTER, receive new weights.
	"""

	# Each worker nodes start at different time.
	time.sleep(2*np.random.rand())

	# Status of current game
	total_score = total_step = 0

	# Send processor name
	comm.send(my_name, dest=MASTER, tag=2)

	# Receive initail weights from master.
	global_weights = comm.recv(source=MASTER, tag=0)
	agent.set_weights(global_weights)

	# Request lastest weights from master every $t_sync training iterations.
	# t : current training iteration.
	t, t_sync = 0, 1

	while True :
		t = (t+1)%t_sync
		sync = (t == 0) # request lastest weights ?

		# Train the model for some game steps.
		# n_step = 5
		n_step = np.random.randint(5, 6)
		(score, n_step, done), loss, raw_grads = agent.train(n_step)
		
		# Update game status.
		total_score+= score
		total_step+= n_step

		# Clipped gradients.
		# grads = [np.clip(x, -100, 100) for x in raw_grads]
		grads = raw_grads

		# Game status.
		stats = {"done":done, "sync":sync}
		if done : # Game is finished.
			# Number of 4-frame steps. How long does he survive ?
			total_step = (total_step + early_skipping)*nb_frames/4.
			stats.update({"score":total_score, "steps": total_step, "loss":loss})

			# Make a new game. Reset game status.
			total_score = total_step = 0

		# Send game status and gradients to master.
		comm.send(stats, dest=MASTER, tag=1)
		sendFLOAT(grads, dest=MASTER, tag=1)

		# Receive lastest weights from master.
		if sync :
			# global_weights = comm.recv(source=MASTER, tag=0)
			recvFLOAT(global_weights, src=MASTER, tag=0)
			agent.set_weights(global_weights)

def controller(network, event) :
	"""
	The brain. Run on MASTER nodes.
	Get gradients from workers, update, send workers lastest weights.
	Run in a thread. Stop when KeyboardInterrupt ($event will be cleared).
	"""
	# Receive workers' names.
	names = ["master"]+[comm.recv(source=i, tag=2) for i in range(1, n_node)]
	print names

	# Send initial weights to workers.
	weights = network.get_weights()
	for x in others :
		comm.send(weights, dest=x, tag=0)

	stt = MPI.Status()
	grads = [np.empty(x.shape, dtype="float32") for x in weights]

	T = 0 # Number of finished games.

	print "Finished :", T, "games."
	bar = Progbar(save_period)

	# Tracking distributed performance.
	nb_update = [0 for x in range(n_node)]
	nb_game = [0 for x in range(n_node)]

	start_time = time.time()

	# Styling.
	color = "\033[45m"
	endc = "\033[0m"

	while event.is_set() :
		# Receive game status and gradients from any workers.
		stats = comm.recv(source=MPI.ANY_SOURCE, tag=1, status=stt)
		src = stt.Get_source()
		recvFLOAT(grads, src=src, tag=1)
		network.apply_gradients(grads)

		nb_update[src]+= 1

		# If worker's game is finished, update status bar and save occasionally.
		if (stats["done"]) :
			T+= 1
			bar.add(1, values=[("score", stats["score"]),
							   ("steps", stats["steps"]),
							   ("loss",  stats["loss"])])
			
			nb_game[src]+= 1

			# Display worker status.
			table_length = (15*4+5)
			print "\n"+" "*table_length
			print "%s%-15s%-15s%-15s%-15s%-5s%s" % \
				(color, "NAME", "UPDS/SEC", "UPDS", "SEC/GAME", "GAME", endc)
			delta_time = time.time() - start_time
			for i in range(1, n_node) :
				upd_per_sec = 1.*nb_update[i]/delta_time
				if (nb_game[i] > 0) :
					sec_per_game = delta_time/nb_game[i]
				else :
					sec_per_game = -1
				print "%-15s%-15.3f%-15i%-15.3f%-15i" % \
					(names[i], upd_per_sec, nb_update[i], sec_per_game, nb_game[i])
			print "-"*table_length
			sum_update = sum(nb_update)
			sum_game = sum(nb_game)
			print "%-15s%-15.3f%-15i%-15.3f%-15i" % \
				("Total", sum_update/delta_time, sum_update, delta_time/sum_game, sum_game)
			print "\033[F"*(n_node+3)+"Time elapsed :",
			timer(start_time, time.time())
			print ", learning rate = %.6f, step = %i"%(K.eval(network.learning_rate), K.eval(network.global_step)),
			print "\033[F",

			if (T%save_period == 0) :
				network.save(SAVE_PATH)
				print "\nFinished :", T, "games.%s" %(" "*15*3)
				bar = Progbar(save_period)

		# If worker needs new weights, send him !
		if (stats["sync"]) :
			weights = network.get_weights()
			sendFLOAT(weights, dest=src, tag=0)

def player(network, event) :
	"""
	Playing and visualizing a game. For human watchers only ! Run on MASTER node.
	Run in a thread. Stop when KeyboardInterrupt ($event will be cleared).
	"""
	while event.is_set() :
		_, _, (V, P) = network.act()
		# print V, P
		network.env.render()
		if (network.env.done) :
			network.reset_game()
		time.sleep(0.1)

def timer(start,end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print "{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)),


#-------------------------------------------------------------------------
if __name__ == "__main__" :
	# This node's environment.
	e = Environment(game_name, nb_frames, preprocess = "default")

	if (my_id == MASTER) :
		print my_name, "is master."
		brain = Brain(e, early_skipping, use_LSTM = LSTM, name="global"+str(my_id))
		brain.model.summary()
		if (LOAD_PATH is not None) :
			brain.load(LOAD_PATH)

		# event is clear => all threads stop.
		event = threading.Event()
		event.set()

		thr = []
		thr.append(threading.Thread(target=controller, args=(brain, event)))
		if (play) :
			thr.append(threading.Thread(target=player, args=(brain, event)))
		for x in thr :
			x.start()
		try :
			while True :
				# Save CPU power.
				time.sleep(0.1)
		except KeyboardInterrupt :
			print "\n"*(n_node+2)
			print "Stop !"
			# Stop all threads.
			event.clear()
			for x in thr :
				x.join()

	else :
		# I'm worker node.
		agent = Agent(e, early_skipping, use_LSTM = LSTM, gamma = 0.99, name="agent"+str(my_id))
		agent.model.summary()
		learner(my_id, agent)