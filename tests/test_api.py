import requests
import six.moves.urllib.parse as urlparse
import os
import time
import pytest

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
    time.sleep(0.25) # give it a moment to settle
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

needs_api_key = pytest.mark.skipif(os.environ.get('OPENAI_GYM_API_KEY') is None, reason="needs OPENAI_GYM_API_KEY")


########## TESTS ##########

##### Valid use cases #####

@with_server
def test_create_destroy():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    assert instance_id in client.env_list_all()
    client.env_close(instance_id)
    assert instance_id not in client.env_list_all()

@with_server
def test_action_space_discrete():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    action_info = client.env_action_space_info(instance_id)
    assert action_info['name'] == 'Discrete'
    assert action_info['n'] == 2

@with_server
def test_action_space_sample():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    action = client.env_action_space_sample(instance_id)
    assert 0 <= action < 2

@with_server
def test_action_space_contains():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    action_info = client.env_action_space_info(instance_id)
    assert action_info['n'] == 2
    assert client.env_action_space_contains(instance_id, 0) == True
    assert client.env_action_space_contains(instance_id, 1) == True
    assert client.env_action_space_contains(instance_id, 2) == False

@with_server
def test_observation_space_box():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    obs_info = client.env_observation_space_info(instance_id)
    assert obs_info['name'] == 'Box'
    assert len(obs_info['shape']) == 1
    assert obs_info['shape'][0] == 4
    assert len(obs_info['low']) == 4
    assert len(obs_info['high']) == 4

@with_server
def test_observation_space_contains():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    obs_info = client.env_observation_space_info(instance_id)
    assert obs_info['name'] == 'Box'
    assert client.env_observation_space_contains(instance_id, {"name":"Box"})
    assert client.env_observation_space_contains(instance_id, {"shape": (4,)})
    assert client.env_observation_space_contains(instance_id, {"name":"Box", "shape": (4,)})

@with_server
def test_reset():
    client = gym_http_client.Client(get_remote_base())

    instance_id = client.env_create('CartPole-v0')
    init_obs = client.env_reset(instance_id)
    assert len(init_obs) == 4

    instance_id = client.env_create('FrozenLake-v0')
    init_obs = client.env_reset(instance_id)
    assert init_obs == 0

@with_server
def test_step():
   client = gym_http_client.Client(get_remote_base())

   instance_id = client.env_create('CartPole-v0')
   client.env_reset(instance_id)
   [observation, reward, done, info] = client.env_step(instance_id, 0)
   assert len(observation) == 4
   assert type(reward) == float
   assert type(done) == bool
   assert type(info) == dict

   instance_id = client.env_create('FrozenLake-v0')
   client.env_reset(instance_id)
   [observation, reward, done, info] = client.env_step(instance_id, 0)
   assert type(observation) == int

@needs_api_key
@with_server
def test_monitor_start_close_upload():
    assert os.environ.get('OPENAI_GYM_API_KEY')
        # otherwise test is invalid

    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    client.env_monitor_start(instance_id, 'tmp', force=True)
    client.env_reset(instance_id)
    client.env_step(instance_id, 1)
    client.env_monitor_close(instance_id)
    client.upload('tmp')

##### API usage errors #####

@with_server
def test_bad_instance_id():
    ''' Test all methods that use instance_id with an invalid ID'''
    client = gym_http_client.Client(get_remote_base())
    try_these = [lambda x: client.env_reset(x),
                 lambda x: client.env_step(x, 1),
                 lambda x: client.env_action_space_info(x),
                 lambda x: client.env_observation_space_info(x),
                 lambda x: client.env_monitor_start(x, directory='tmp', force=True),
                 lambda x: client.env_monitor_close(x),
                 lambda x: client.env_close(x)]
    for call in try_these:
        try:
            call('bad_id')
        except gym_http_client.ServerError as e:
            assert 'Instance_id' in e.message
            assert e.status_code == 400
        else:
            assert False

@with_server
def test_missing_param_env_id():
    ''' Test client failure to provide JSON param: env_id'''
    class BadClient(gym_http_client.Client):
        def env_create(self, env_id):
            route = '/v1/envs/'
            data = {} # deliberately omit env_id
            resp = self._post_request(route, data)
            instance_id = resp.json()['instance_id']
            return instance_id
    client = BadClient(get_remote_base())
    try:
        client.env_create('CartPole-v0')
    except gym_http_client.ServerError as e:
        assert 'env_id' in e.message
        assert e.status_code == 400
    else:
        assert False

@with_server
def test_missing_param_action():
    ''' Test client failure to provide JSON param: action'''
    class BadClient(gym_http_client.Client):
        def env_step(self, instance_id, action):
            route = '/v1/envs/{}/step/'.format(instance_id)
            data = {} # deliberately omit action
            resp = self._post_request(route, data)
            observation = resp.json()['observation']
            reward = resp.json()['reward']
            done = resp.json()['done']
            info = resp.json()['info']
            return [observation, reward, done, info]
    client = BadClient(get_remote_base())

    instance_id = client.env_create('CartPole-v0')
    client.env_reset(instance_id)
    try:
        client.env_step(instance_id, 1)
    except gym_http_client.ServerError as e:
        assert 'action' in e.message
        assert e.status_code == 400
    else:
        assert False

@with_server
def test_missing_param_monitor_directory():
    ''' Test client failure to provide JSON param: directory'''
    class BadClient(gym_http_client.Client):
        def env_monitor_start(self, instance_id, directory,
                              force=False, resume=False):
            route = '/v1/envs/{}/monitor/start/'.format(instance_id)
            data = {'force': force,
                'resume': resume} # deliberately omit directory
            self._post_request(route, data)
    client = BadClient(get_remote_base())

    instance_id = client.env_create('CartPole-v0')
    try:
        client.env_monitor_start(instance_id, 'tmp', force=True)
    except gym_http_client.ServerError as e:
        assert 'directory' in e.message
        assert e.status_code == 400
    else:
        assert False

@needs_api_key
@with_server
def test_missing_param_upload_directory():
    ''' Test client failure to provide JSON param: directory'''
    class BadClient(gym_http_client.Client):
        def upload(self, training_dir, algorithm_id=None, api_key=None):
            if not api_key:
                api_key = os.environ.get('OPENAI_GYM_API_KEY')

            route = '/v1/upload/'
            data = {'algorithm_id': algorithm_id,
                    'api_key': api_key}
                # deliberately omit training_dir
            self._post_request(route, data)
    client = BadClient(get_remote_base())

    assert os.environ.get('OPENAI_GYM_API_KEY')
        # otherwise test is invalid

    instance_id = client.env_create('CartPole-v0')
    client.env_monitor_start(instance_id, 'tmp', force=True)
    client.env_reset(instance_id)
    client.env_step(instance_id, 1)
    client.env_monitor_close(instance_id)
    try:
        client.upload('tmp')
    except gym_http_client.ServerError as e:
        assert 'training_dir' in e.message
        assert e.status_code == 400
    else:
        assert False

@with_server
def test_empty_param_api_key():
    ''' Test client failure to provide non-empty JSON param: api_key'''
    class BadClient(gym_http_client.Client):
        def upload(self, training_dir, algorithm_id=None, api_key=None):
            route = '/v1/upload/'
            data = {'algorithm_id': algorithm_id,
                    'training_dir': 'tmp',
                    'api_key': ''} # deliberately empty string
            self._post_request(route, data)
    client = BadClient(get_remote_base())
    instance_id = client.env_create('CartPole-v0')
    client.env_monitor_start(instance_id, 'tmp', force=True)
    client.env_reset(instance_id)
    client.env_step(instance_id, 1)
    client.env_monitor_close(instance_id)
    try:
        client.upload('tmp')
    except gym_http_client.ServerError as e:
        assert 'api_key' in e.message
        assert e.status_code == 400
    else:
        assert False


##### Gym-side errors #####

@with_server
def test_create_malformed():
    client = gym_http_client.Client(get_remote_base())
    try:
        client.env_create('bad string')
    except gym_http_client.ServerError as e:
        assert 'malformed environment ID' in e.message
        assert e.status_code == 400
    else:
        assert False

# @with_server
# def test_missing_API_key():
#    client = gym_http_client.Client(get_remote_base())
#    cur_key = os.environ.get('OPENAI_GYM_API_KEY')
#    os.environ['OPENAI_GYM_API_KEY'] = ''
#    try:
#        print 'UPLOADING'
#        print cur_key
#        client.upload('tmp')
#        print '*****'
#    except requests.HTTPError, e:
#        assert e.response.status_code == 400
#    else:
#        assert False
#    finally:
#        if cur_key:
#            os.environ['OPENAI_GYM_API_KEY'] = cur_key
