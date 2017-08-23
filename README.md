# A3C
Async Advantage Actor Critic

Implementation of https://arxiv.org/abs/1602.01783 using Tensorflow - Keras and OpenAI gym environments.  

Lazy guide - how to run :
1) Install Python, Tensorflow, Keras, OpenaI gym and MPI.
2) If you want to run this code on a cluster, [configurate](http://mpitutorial.com/tutorials/) MPI so that it can run on your cluster. 
3) Edit file `peers` to specify nodes' IP addresses.
4) Command line : `mpiexec -np 12 -hostfile peers python -B main.py`. Replace `12` with `number of agent + 1`.
5) Edit `main.py` constants to specify which environment to learn, where to save, where to load,...
6) Edit `Model.py` to change the network architecture.

How it works :
1) There's a central master machine which stores a CNN.
2) Each worker machine stores its own CNN agent(s) playing their own game.
3) Initially each agents is a copy of the master's CNN (ie. same weights).
4) Each agent plays games, calculates the gradient, sends it to the master and waits for response.
5) The master receives gradient, updates its CNN using its optimizer (ie. RMSprop, Adam...), sends a new copy of its CNN to the corresponding agent and displays current progress to human user.
6) The agent receives response (new weights) from master and replace it's CNN weights with new weights.
7) Master and agents run that loop forever, or until Ctrl + C is hit.

Computation can be distributed on multi machine using MPI. This only makes sense when communication cost (which is proportional to size of gradient tensors) is smaller then back-propagation cost plus environment simulation cost.  
This implies that :
1) a big network with small amount of parameters like Convolutional layers
2) or heavy environments like GTA V, Starcraft 2...
... would benefit from distributed computing.
