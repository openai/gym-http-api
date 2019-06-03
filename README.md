Gym domains with the command line
============

This experimental repository utilizes Parsers to execute various reinforcement learning algorithms in appropriate Gym domains. For the time being, five algorithms are supported (deep Q-learning network, 
asynchronous advantage actor-critic, policy gradient, deep deterministic policy gradient, and soft-actor critic), and one may run two separate Main files according to whether or not the domains that certain 
algorithms use are discrete or continuous (Main_disc.py and Main_con.py, respectively).

This repository imports Gym domains that are not within the general repository, such as Atari and Retro, for our eventual testing with said domains.


Installation
============

To download the code and install the requirements, run the following prompt commands:

    git clone https://github.com/nazaruka/gym-http-api
    cd gym-http-api
	conda install -c conda-forge swig (in Anaconda Prompt!)
    pip install -r requirements.txt


Implementation
============

Algorithms are shortened according to the following acronymic forms:

- Deep Q-learning network: DQN
- Asynchronous advantage actor-critic: A3C
- Policy gradient: PG
- Deep deterministic policy gradient: DDPG
- Soft actor-critic: SAC

Firstly, decide between Main_disc.py or Main_con.py, depending on which algorithms and environments you would like to test.
NB: Main_disc.py will take A3C, DQN, and PG; Main_con.py will take DDPG and SAC.

Next, open your chosen file according to the following call format:

python Main_disc.py/Main_con.py [-h] -a ALG -d DOMAIN [-e EPSILON] [--estep EPSILON_STEP] [--edecay EPSILON_DECAY] [--emin EPSILON_MIN] [-g GAMMA] [-t TAU] [--nepisode NEPISODE]

  -h, --help            				Show help message and exit
  -a ALG                				The algorithm you wish to work with.
  -d DOMAIN          				The OpenAI Gym domain you wish to work in.
  -e EPSILON          		 		DQN: epsilon parameter (default 0.2)
  --estep EPSILON_STEP			DQN: epsilon step (default 10)
  --edecay EPSILON_DECAY		DQN: epsilon decay (default 0.99)
  --emin EPSILON_MIN    		DQN: minimum epsilon (default 0.001)
  -g GAMMA              				Gamma parameter (default 0.999)
  -t TAU                					Tau parameter (default 0.995)
  --nepisode NEPISODE   		Number of episodes (default 1000)
  
Several of these arguments are limited to specific algorithms; all save for ALG and DOMAIN are optional.

Here is a list of appropriate Gym domains to run based on the domains' distributions:

- Main_disc (for DQN, A3C, PG): Acrobot-v1, CartPole-v1, MountainCar-v0, LunarLander-v2
- Main_con (for DDPG, SAC): Pendulum-v1


Contributors
============

Assembled by Aleksandr "Alex" Nazaruk '22 and Dr. Jacob Schrum of Southwestern University. 
We give credit to the original Uber AI repository, as well as to the following repository for providing much of the basis for our code: https://github.com/sarcturus00/Tidy-Reinforcement-learning