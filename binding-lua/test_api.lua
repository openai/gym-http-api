local function getTest(opt)
   local gymClient = opt.gymClient
   local verbose = opt.verbose
   local render = opt.render
   local video_callable = opt.video_callable
   
   -- debug mode, verbose output 
   if verbose == false then
        oldPrint = print
        print = function() end
   end

   local HttpClient = require("httpclient")
   local function runTest(env_id)
      require('math')
      -- Set up client
      local base = 'http://127.0.0.1:5000'
      local client = gymClient.new(base)
      print('Creation of client, server at ' .. base)
      -- Set up environment
      print('Creation of ' .. env_id)
      local instance_id = client:env_create(env_id)
      print('Created ' .. env_id)
      print('****************************************************')
      print('Get action space')
      local action_space = client:env_action_space_info(instance_id)
      print('Action space:')
      print(action_space)
      print('****************************************************')
      print('Get observation space')
      local observation_space = client:env_observation_space_info(instance_id)
      print('Obervation space:')
      print(observation_space)
      print('****************************************************')
      -- Run random experiment with monitor
      local outdir = '/tmp/random-agent-results'
      local resume = false
      local force = true
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
      local obs = client:env_reset(instance_id)
      print('Environment reset')
      print('****************************************************')
      print('Attempt sample action space:')
      local action = client:env_action_space_sample(instance_id)
      print('Action:')
      print(action)
      print('****************************************************')
      print('Attempt step in environment:')
      local ob, reward, done, info = client:env_step(instance_id, action, render)
      print('Success')
      print('State:')
      print(ob)
      print('Reward: ' .. reward)
      print('Done: ')
      print(done)
      print('Info: ')
      print('****************************************************')
      local episode_count = 2
      local max_steps = 5
      local reward = 0
      local done = False
      print('Setting up experiment with following configuration:')
      print('Episodes:' .. episode_count)
      print('Max steps:' .. max_steps)
      for i = 1,episode_count do
         obs = client:env_reset(instance_id)
         for j = 1,max_steps do
            action = client:env_action_space_sample(instance_id)
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
   return runTest
end
return getTest