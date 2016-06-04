import requests
import urlparse
import json
import os

class Client(object):
    """
    Gym client to interface with gym_http_server
    """
    def __init__(self, remote_base):
        self.remote_base = remote_base

    def _post_request(self, route, data):
        headers = {'Content-type': 'application/json'}
        resp = requests.post(urlparse.urljoin(self.remote_base, route),
                            data=json.dumps(data),
                            headers=headers)
        resp.raise_for_status()
        return resp

    def _get_request(self, route):
        resp = requests.get(urlparse.urljoin(self.remote_base, route))
        resp.raise_for_status()
        return resp
        
    def env_create(self, env_id):
        route = '/v1/envs/'
        data = {'env_id': env_id}
        resp = self._post_request(route, data)
        instance_id = resp.json()['instance_id']
        return instance_id

    def env_list_all(self):
        route = '/v1/envs/'
        resp = self._get_request(route)
        all_envs = resp.json()['all_envs']
        return all_envs

    def env_check_exists(self, instance_id):
        route = '/v1/envs/{}/check_exists/'.format(instance_id)
        resp = self._post_request(route, None)
        exists = resp.json()['exists']
        return exists

    def env_reset(self, instance_id):
        route = '/v1/envs/{}/reset/'.format(instance_id)
        resp = self._post_request(route, None)
        observation = resp.json()['observation']
        return observation

    def env_step(self, instance_id, action):    
        route = '/v1/envs/{}/step/'.format(instance_id)
        data = {'action': action}
        resp = self._post_request(route, data)
        observation = resp.json()['observation']
        reward = resp.json()['reward']
        done = resp.json()['done']
        info = resp.json()['info']
        return [observation, reward, done, info]

    def env_action_space_info(self, instance_id):
        route = '/v1/envs/{}/action_space/'.format(instance_id)
        resp = self._get_request(route)
        info = resp.json()['info']
        return info

    def env_observation_space_info(self, instance_id):
        route = '/v1/envs/{}/observation_space/'.format(instance_id)
        resp = self._get_request(route)
        info = resp.json()['info']
        return info

    def env_monitor_start(self, instance_id, directory,
                              force=False, resume=False):
        route = '/v1/envs/{}/monitor/start/'.format(instance_id)
        data = {'directory': directory,
                'force': force,
                'resume': resume}
        self._post_request(route, data)

    def env_monitor_close(self, instance_id):
        route = '/v1/envs/{}/monitor/close/'.format(instance_id)
        self._post_request(route, None)

    def upload(self, training_dir, algorithm_id=None, writeup=None, 
                   api_key=None, ignore_open_monitors=False):
        if not api_key:
            api_key = os.environ.get('OPENAI_GYM_API_KEY')

        route = '/v1/upload/'
        data = {'training_dir': training_dir,
                'algorithm_id': algorithm_id,
                'writeup': writeup,
                'api_key': api_key,
                'ignore_open_monitors': ignore_open_monitors}
        self._post_request(route, data)

    def shutdown_server(self):
        route = '/v1/shutdown/'
        self._post_request(route, None)

if __name__ == '__main__':
    remote_base = 'http://127.0.0.1:5000'
    client = Client(remote_base)

    # Create environment
    env_id = 'CartPole-v0'
    instance_id = client.env_create(env_id)

    # Check properties
    exists = client.env_check_exists(instance_id)
    all_envs = client.env_list_all()
    action_info = client.env_action_space_info(instance_id)
    obs_info = client.env_observation_space_info(instance_id)

    # Run a single step
    client.env_monitor_start(instance_id, directory='tmp', force=True)
    init_obs = client.env_reset(instance_id)
    [observation, reward, done, info] = client.env_step(instance_id, 1)
    client.env_monitor_close(instance_id)
    client.upload(training_dir='tmp')

    


