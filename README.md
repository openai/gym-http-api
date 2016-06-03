gym-http-api
******

This project provides a local REST API to the [gym](https://github.com/openai/gym) open-source library, allowing development in languages other than python.

A python client is included, to demonstrate how to interact with the server. Contributions of clients in other languages are welcomed!


Installation
============

To download the code and install the requirements, you can run the following shell commands:

.. code:: shell
		  git clone https://github.com/catherio/gym-http-api
		  cd gym-http-api
		  pip install -r requirements.txt


Getting started
============

This code is intended to be run locally. The server runs in python. A demo client written in python is provided, but you can implement your own clients using any language.

To start the server from the command line:

.. code:: shell
	python gym_server.py


To launch a demo client in python, and test it out by creating a new environment:

.. code:: python
    remote_base = 'http://127.0.0.1:5000'
    client = Client(remote_base)

    env_id = 'CartPole-v0'
    instance_id = client.env_create(env_id)
    exists = client.env_check_exists(instance_id)


API documentation
============

TODO fully document the API
