import nose2
import requests
import urlparse

from threading import Thread

import gym_http_server
import gym_http_client

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########## CONFIGURATION ##########

host = '127.0.0.1'
port = '5000'
def get_remote_base():
    return 'http://{host}:{port}'.format(host=host, port=port)

def setup_background_server():
    def start_server(app):
        app.run(threaded=True, host=host, port=port)

    global server_thread
    server_thread = Thread(target=start_server,
                       args=(gym_http_server.app,))
    server_thread.daemon = True
    server_thread.start()
    logger.info('Server setup complete')

def teardown_background_server():
    route = '/v1/shutdown/'
    headers = {'Content-type': 'application/json'}
    requests.post(urlparse.urljoin(get_remote_base(), route),
                  headers=headers)
    server_thread.join() # wait until teardown happens
    logger.info('Server teardown complete')

def with_server(fn):
    fn.setup = setup_background_server
    fn.teardown = teardown_background_server
    return fn

########## Tests ##########

@with_server
def test_create():
    client = gym_http_client.Client(get_remote_base())
    client.env_create('CartPole-v0')

@with_server
def test_create_malformed():
    remote_base = 'http://127.0.0.1:5000'
    client = gym_http_client.Client(get_remote_base())
    try:
        client.env_create('bad string')
    except requests.HTTPError, e:
        assert(e.response.status_code == 400)
    else:
        assert False
