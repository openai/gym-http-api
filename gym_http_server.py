from flask import Flask, request, jsonify
import uuid
import gym

class Envs(object):
    """
    Container and manager for the environments instantiated on this server.

    When a new environment is created, such as with envs.create('CartPole-v0'), it is stored under a short identifier (such as '3c657dbc'). Future API calls make use of this instance_id to identify which environment should be manipulated.
    """
    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def create(self, env_id):
        env = gym.make(env_id)
        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def check_exists(self, instance_id):
        return instance_id in self.envs

    def reset(self, instance_id):
        env = self.envs[instance_id]
        obs = env.reset()
        return env.observation_space.to_jsonable(obs)

    def step(self, instance_id, action):
        env = self.envs[instance_id]
        action_from_json = int(env.action_space.from_jsonable(action))
        [observation, reward, done, info] = env.step(action_from_json)
        obs_jsonable = env.observation_space.to_jsonable(observation)
        return [obs_jsonable, reward, done, info]
    
    def monitor_start(self, instance_id, directory, force, resume):
        env = self.envs[instance_id]
        env.monitor.start(directory, force=force, resume=resume)

    def monitor_close(self, instance_id):
        env = self.envs[instance_id]
        env.monitor.close()

app = Flask(__name__)
envs = Envs()

@app.route('/v1/envs/', methods=['POST'])
def env_create():
    """
    Instantiates an instance of the specified environment
    
    Parameters:
        - env_id: gym environment ID string, such as 'CartPole-v0'
    Returns:
        - instance_id: a short identifier (such as '3c657dbc') for the created environment instance. The instance_id is used in future API calls to identify the environment to be manipulated
    """

    env_id = request.get_json()['env_id']
    instance_id = envs.create(env_id)
    return jsonify(instance_id = instance_id)

@app.route('/v1/envs/<instance_id>/check_exists/', methods=['POST'])
def env_check_exists(instance_id):
    """
    Determines whether the specified instance_id corresponds to a valid environment instance that has been created.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc') for the environment instance
    Returns:
        - exists: True or False, indicating whether the given instance exists
    """
    exists = envs.check_exists(instance_id)
    return jsonify(exists = exists)

@app.route('/v1/envs/<instance_id>/reset/', methods=['POST'])
def env_reset(instance_id):
    """
    Resets the state of the environment and returns an initial observation.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc') for the environment instance
    Returns:
        - observation: the initial observation of the space
    """  
    observation = envs.reset(instance_id)
    return jsonify(observation = observation)

@app.route('/v1/envs/<instance_id>/step/', methods=['POST'])
def env_step(instance_id):
    """
    Run one timestep of the environment's dynamics.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc') for the environment instance
        - action: an action provided by the environment
    Returns:
        - observation: agent's observation of the current environment
        - reward: amount of reward returned after previous action
        - done: whether the episode has ended
        - info: a dict containing auxiliary diagnostic information
    """  
    action = request.get_json()['action']
    [obs_jsonable, reward, done, info] = envs.step(instance_id, action)
    return jsonify(observation = obs_jsonable,
                    reward = reward, done = done, info = info)

@app.route('/v1/envs/<instance_id>/monitor/start/', methods=['POST'])
def env_monitor_start(instance_id):
    """
    Start monitoring.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc') for the environment instance
        - force (default=False): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.")
        - resume (default=False): Retain the training data already in this directory, which will be merged with our new data
    
    (NOTE: the 'video_callable' parameter from the native env.monitor.start function is NOT implemented)
    """  
    request_data = request.get_json()

    directory = request_data['directory']
    force = request_data.get('force', False)
    resume = request_data.get('resume', False)

    envs.monitor_start(instance_id, directory, force, resume)
    # NOTE: no video_callable implemented yet
    return ('', 204)

@app.route('/v1/envs/<instance_id>/monitor/close/', methods=['POST'])
def env_monitor_close(instance_id):
    """
    Flush all monitor data to disk.
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc') for the environment instance
    """
    envs.monitor_close(instance_id)
    return ('', 204)

@app.route('/v1/upload/', methods=['POST'])
def upload():
    """
    Upload the results of training (as automatically recorded by your env's monitor) to OpenAI Gym.
    
    Parameters:
        - training_dir: A directory containing the results of a training run.
        - algorithm_id (default=None): An arbitrary string indicating the paricular version of the algorithm (including choices of parameters) you are running.
        - writeup (default=None): A Gist URL (of the form https://gist.github.com/<user>/<id>) containing your writeup for this evaluation.
        - api_key (default=None): Your OpenAI API key
    """  
    request_data = request.get_json()

    training_dir = request_data['training_dir']
    algorithm_id = request_data.get('algorithm_id', None)
    writeup = request_data.get('writeup', None)
    api_key = request_data.get('api_key', None)
    ignore_open_monitors = request_data.get('ignore_open_monitors', False)

    gym.upload(training_dir, algorithm_id, writeup, api_key,
                   ignore_open_monitors)

if __name__ == '__main__':
    app.run()

