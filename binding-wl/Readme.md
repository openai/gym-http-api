# Wolfram Language Interface to the OpenAI Gym

Usage of this package assumes the OpenAI Gym (https://github.com/openai/gym) is installed, and the Gym HTTP Server (https://github.com/openai/gym-http-api) is running.

## General Information
Load the `GymEnvironment` package:
```
Get["path/to/gym_http_client.wl"]
```
Once the package is loaded, a list of all exposed functions can be obtained via:
```
Names["GymEnvironment`*"]
```
Obtain documentation for a specific package function:
```
?EnvCreate
```

## Running the Example Agent
For an example on how to use this package, see the Wolfram Script `example_agent.wl`.
This can be run via:
```
wolframscript -script path/to/example_agent.wl
```

