<img align="left" src="http://i.imgur.com/568Luwb.png">gym-http-api
============

This project provides a local REST API to the [gym](https://github.com/openai/gym) open-source library, allowing development in languages other than python.

A python client is included, to demonstrate how to interact with the server.

Additional languages:
  * C++ (within this repository, author: Oleg Klimov)
  * lua (within this repository, work-in-progress)
  * Scala (in progress, author: Flavio Truzzi)
  * ... Contributions of clients in other languages are welcomed!

Installation
============

To download the code and install the requirements, you can run the following shell commands:

    git clone https://github.com/catherio/gym-http-api
    cd gym-http-api
    pip install -r requirements.txt


Getting started
============

This code is intended to be run locally by a single user. The server runs in python. You can implement your own HTTP clients using any language; a demo client written in python is provided to demonstrate the idea.

To start the server from the command line, run this:

    python gym_http_server.py
    
In a separate terminal, you can then try running the example python agent and see what happens:

    python example_agent.py  

The example lua agent behaves very similarly:

    cd binding-lua
    lua example_agent.lua

You can also write code like this to create your own client, and test it out by creating a new environment. For example, in python:

    remote_base = 'http://127.0.0.1:5000'
    client = Client(remote_base)

    env_id = 'CartPole-v0'
    instance_id = client.env_create(env_id)
    client.env_step(instance_id, 0)


Testing
============

This repository contains integration tests, using the python client implementation to send requests to the local server. They can be run using the `nose2` framework. From a shell (such as bash) you can run nose2 directly:

    cd gym-http-api
    nose2


API specification
============

  * POST `/v1/envs/`
      * Create an instance of the specified environment
      * param: `env_id` -- gym environment ID string, such as 'CartPole-v0'
      * returns: `instance_id` -- a short identifier (such as '3c657dbc')
	    for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated

  * GET `/v1/envs/`
      * List all environments running on the server
	  * returns: `envs` -- dict mapping `instance_id` to `env_id` 
	    (e.g. `{'3c657dbc': 'CartPole-v0'}`) for every env on the server

  * POST `/v1/envs/<instance_id>/reset/`
      * Reset the state of the environment and return an initial
        observation.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
      * returns: `observation` -- the initial observation of the space
    
  * POST `/v1/envs/<instance_id>/step/`
      * Reset the state of the environment and return an initial
        observation.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
	  * param: `action` -- an action to take in the environment
      * returns: `observation` -- agent's observation of the current
        environment
      * returns: `reward` -- amount of reward returned after previous action
      * returns: `done` -- whether the episode has ended
      * returns: `info` -- a dict containing auxiliary diagnostic information

  * GET `/v1/envs/<instance_id>/action_space/`
      * Get information (name and dimensions/bounds) of the env's
        `action_space`
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance  
      * returns: `info` -- a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space

  * GET `/v1/envs/<instance_id>/observation_space/`
      * Get information (name and dimensions/bounds) of the env's
        `observation_space`
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance  
      * returns: `info` -- a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space

  * POST `/v1/envs/<instance_id>/monitor/start/`
      * Start monitoring
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance  
      * param: `force` (default=False) -- Clear out existing training
        data from this directory (by deleting every file
        prefixed with "openaigym.")
      * param: `resume` (default=False) -- Retain the training data
        already in this directory, which will be merged with
        our new data
      * (NOTE: the `video_callable` parameter from the native
    `env.monitor.start` function is NOT implemented)

  * POST `/v1/envs/<instance_id>/monitor/close/`
      * Flush all monitor data to disk
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance 

  * POST `/v1/upload/`
      * Flush all monitor data to disk
      * param: `training_dir` -- A directory containing the results of a
        training run.
      * param: `api_key` -- Your OpenAI API key
      * param: `algorithm_id` (default=None) -- An arbitrary string
        indicating the paricular version of the algorithm
        (including choices of parameters) you are running.
   
  * POST `/v1/shutdown/`
      * Request a server shutdown
      * Currently used by the integration tests to repeatedly create and destroy fresh copies of the server running in a separate thread

TODOs
===============

python TODOs
- Make the directory structure better conform to standard python package structures (http://www.kennethreitz.org/essays/repository-structure-and-python, http://peterdowns.com/posts/first-time-with-pypi.html)
- Implement 'sample' (and test it)
- Implement 'contains' (and test it)
- Docker integration
- Handle ResetNeeded while monitor is active
- Handle APIConnectionError: Unexpected error communicating with OpenAI Gym
- Check: was anything improved by adding session / socket reuse?
- Reports of "broken pipe" errors: reproduce and investigate
- Measure latency/performance. How slow is HTTP, Flask? What is the use case for this implementation, versus a potential future faster ZeroMQ implementation?
- Make remote environments have the same interface as non-remote environments
- Get Travis CI working
- Test all possible environments in integration tests
- Handle the error thrown if the directory isn't cleared for the monitor
- Document the fact that two-monitors-open will cause a problem; be clear that this is meant to be one-client

lua client wishlist:
- Error handling, similar to what the python client currently has
- Implement the ability to set "render=True"

Contributors
============
  * Catherine Olsson
  * Jie Tang
  * Greg Brockman
  * Flavio Truzzi
  * Oleg Klimov
  * Jess Smith
  * Kory Mathewson
  * Leonardo Araujo dos Santos
  * Paul Anton
  * Ruben Fiszel
  * Niven Achenjang
