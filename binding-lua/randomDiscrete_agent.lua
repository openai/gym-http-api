local GymClient = require("gym_http_client")

-- Set up client
base = 'http://127.0.0.1:5000'
local client = GymClient.new(base)

-- Set up environment
env_id = 'CartPole-v0'
instance_id = client:env_create(env_id)

-- Run random experiment with monitor
outdir = '/tmp/random-agent-results'
client:env_monitor_start(instance_id, outdir, true)
render = false

episode_count = 100
max_steps = 500
reward = 0
done = False

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

-- Dump result info to disk
client:env_monitor_close(instance_id)

-- Upload to the scoreboard. 'OPENAI_GYM_API_KEY' set on the client side
client:upload(outdir)