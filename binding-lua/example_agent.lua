require('math')
local GymClient = require("gym_http_client")
local HttpClient = require("httpclient")

-- Set up client
base = 'http://127.0.0.1:5000'
local client = GymClient.new(base)

-- Set up environment
env_id = 'Pendulum-v0'
instance_id = client:env_create(env_id)

action_space = client:env_action_space_info(env_id)

-- Run random experiment with monitor
outdir = '/tmp/random-agent-results'
video_callable = false
resume = false
force = true
client:env_monitor_start(instance_id, outdir, force, resume, video_callable)
render = true

episode_count = 100
max_steps = 200
reward = 0
done = False

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

-- Dump result info to disk
client:env_monitor_close(instance_id)

-- Upload to the scoreboard. This expects the 'OPENAI_GYM_API_KEY'
-- environment variable to be set on the client side.
client:upload(outdir)
