require('math')
local GymClient = require("gym_http_client")
local HttpClient = require("httpclient")

local function runTest(env_id)
	-- Set up client
	base = 'http://127.0.0.1:5000'
	local client = GymClient.new(base)
	print('Creation of client, server at ' .. base)
	-- Set up environment
	print('Creation of ' .. env_id)
	instance_id = client:env_create(env_id)
	print('Created ' .. env_id)
	print('****************************************************')
	print('Get action space')
	action_space = client:env_action_space_info(instance_id)
	print('Action space:')
	print(action_space)
	print('****************************************************')
	print('Get observation space')
	observation_space = client:env_observation_space_info(instance_id)
	print('Obervation space:')
	print(observation_space)
	print('****************************************************')
	-- Run random experiment with monitor
	outdir = '/tmp/random-agent-results'
	video_callable = false
	resume = false
	force = true
	render = true
	print('Start monitor:')
	client:env_monitor_start(instance_id, outdir, force, resume, video_callable)
	print('Connected to monitor')
	print('InstanceID:' .. instance_id)
	print('Output Directory:' .. outdir)
	print('Force overwrite: ')
	print(force)
	print('Resume: ')
	print(resume)
	print('Video:')
	print(video_callable)
	print('****************************************************')
	print('Attempt environment reset:')
	obs = client:env_reset(instance_id)
	print('Environment reset')
	print('****************************************************')
	print('Attempt sample action space:')
	action = client:env_action_space_sample(instance_id)
	print('Action:')
	print(action)
	print('****************************************************')
	print('Attempt step in environment:')
	ob, reward, done, info = client:env_step(instance_id, action, render)
	print('Success')
	print('State:')
	print(ob)
	print('Reward: ' .. reward)
	print('Done: ')
	print(done)
	print('Info: ')
	print('****************************************************')
	episode_count = 10
	max_steps = 20
	reward = 0
	done = False
	print('Setting up experiment with following configuration:')
	print('Episodes:' .. episode_count)
	print('Max steps:' .. max_steps)
	for i = 1,episode_count do
	   obs = client:env_reset(instance_id)
	   for j = 1,max_steps do
		  	action = math.random(0, 1)
	      ob, reward, done, info = client:env_step(instance_id, action, render)
	      if done then
				break
		  	end
	   end
	end
	print('Experiment complete.')
	print('****************************************************')
	-- Dump result info to disk
	print('Close monitor and save data to disk:')
	client:env_monitor_close(instance_id)
	print('Monitor closed')
	print('****************************************************')
	-- Upload to the scoreboard. This expects the 'OPENAI_GYM_API_KEY'
	-- environment variable to be set on the client side.
	print('Attempt upload:')
	client:upload(outdir)
	print('Upload successful, testing complete.')
	print('****************************************************')
	return true
end

local testEnvs = {'CartPole-v0', 'FrozenLake-v0'}
for i = 1,#testEnvs do
	local _ = runTest(testEnvs[i])
end