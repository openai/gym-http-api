# gym-http-api: JavaScript Binding


## Building

```
npm install
gulp
```

## Example

This should be run from the `binding-js` directory.

```javascript
var gym = require("./dist/lib/gymHTTPClient")
var client = new gym.default("http://127.0.0.1:5000")

var p = client.envCreate("CartPole-v0");
p.then((reply) => console.log("Reply: " + JSON.stringify(reply)))
p.catch((error) => console.log("Error : " + error))
```

After building the library, you can also run the example agent with `node dist/examples/exampleAgent.js`. 

## API Reference

### General Methods

* __constructor(remote)
* _parseServerErrors(promise)
* _buildURL(route)
* _post(route, data)
* _get(route)

### envCreate(envID)

* **route:** POST `/v1/envs/`
* **description:** Create an instance of the specified environment
* **param:** `env_id`
  * gym environment ID string, such as `'CartPole-v0'`
* **returns:** `instance_id`
  * a short identifier (such as `'3c657dbc'`) for the created environment instance. The instance_id is used in future API calls to identify the environment to be manipulated

### envListAll()

* **route:** GET `/v1/envs/`
* **description:** List all environments running on the server
* **returns:** `envs`
  * dict mapping `instance_id` to `env_id` (e.g. `{'3c657dbc': 'CartPole-v0'}`) for every env on the server

### envReset(instanceID)

* **route:** POST `/v1/envs/<instance_id>/reset/`
* **description:** Reset the state of the environment and return an initial
  observation.
* **param:** `instance_id`
  * a short identifier (such as `'3c657dbc'`) for the environment instance
* **returns:** `observation` -- the initial observation of the space

### envStep(instanceID, action, render)

* **route:** POST `/v1/envs/<instance_id>/step/`
* **description:**  Step though an environment using an action.
* **param:** `instance_id`
  * a short identifier (such as `'3c657dbc'`)
  for the environment instance
* **param:** `action`
  * an action to take in the environment
* **returns:** `observation`
  * agent's observation of the current
  environment
* **returns:** `reward`
  * amount of reward returned after previous action
* **returns:** `done`
  * whether the episode has ended
* **returns:** `info`
  * a dict containing auxiliary diagnostic information

### envActionSpaceInfo(instanceID)

* **route:** GET `/v1/envs/<instance_id>/action_space/`
* **description:** Get information (name and dimensions/bounds) of the environments
  `action_space`
* **param:** `instance_id`
  * a short identifier (such as `'3c657dbc'`) for the environment instance
* **returns:** `info`
  * a dict containing 'name' (such as 'Discrete'), and additional dimensional info (such as 'n') which varies from space to space

### envObservationSpaceInfo(instanceID)

* **route:** GET `/v1/envs/<instance_id>/observation_space/`
* **description:** Get information (name and dimensions/bounds) of the environments
  `observation_space`
* **param:** `instance_id` 
  * a short identifier (such as `'3c657dbc'`)
  for the environment instance
* **returns:** `info` 
  * a dict containing 'name' (such as 'Discrete'), and additional dimensional info (such as 'n') which varies from space to space

### envMonitorStart(instanceID, directory, force, resume)

* **route:** POST `/v1/envs/<instance_id>/monitor/start/`
* **description:** Start monitoring
* **param:** `instance_id`
  * a short identifier (such as `'3c657dbc'`) for the environment instance
* **param:** `force` (default=False)
  * Clear out existing training data from this directory (by deleting every file prefixed with `"openaigym."`)
* **param:** `resume` (default=False)
  * Retain the training data already in this directory, which will be merged with our new data

> NOTE: the `video_callable` parameter from the native
`env.monitor.start` function is NOT implemented

## envMonitorClose(instanceID)

* **route:** POST `/v1/envs/<instance_id>/monitor/close/`
* **description:** Flush all monitor data to disk
* **param:** `instance_id`
  * a short identifier (such as `'3c657dbc'`) for the environment instance

### envClose(instanceID)

* **route:** POST `/v1/envs/<instance_id>/close/`
* **description:** Flush all monitor data to disk
* **param:** `instance_id`
  * a short identifier (such as `'3c657dbc'`) for the environment instance

### upload(trainingDir, algorithmID, apiKey)

* **route:** POST `/v1/upload/`
* **description:** Flush all monitor data to disk
* **param:** `training_dir`
  * A directory containing th results of a training run.
* **param:** `api_key`
  * Your OpenAI API key
* **param:** `algorithm_id` (default=None)
  * An arbitrary string indicating the particular version of the algorithm(including choices of parameters) you are running.

### shutdownServer(self)

* **route:** POST `/v1/shutdown/`
* **description:** Request a server shutdown

> Currently used by the integration tests to repeatedly create and destroy fresh copies of the server running in a separate thread
