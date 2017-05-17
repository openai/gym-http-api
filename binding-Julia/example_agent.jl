# Pkg.clone("https://github.com/JuliaML/OpenAIGymAPI.jl.git")
using OpenAIGymAPI

remote_base = "http://127.0.0.1:5000"
client = GymClient(remote_base)
print(client)

# Create environment
env_id = "CartPole-v0"
instance_id = env_create(client, env_id)
print(instance_id)

# List all environments
all_envs = env_list_all(client)
print(all_envs)

# Set up agent
action_space_info = env_action_space_info(client, instance_id)
print(action_space_info)
agent = action_space_info["n"] # perform same action every time

# Run experiment, with monitor
outdir = "/tmp/random-agent-results"
env_monitor_start(client, instance_id, outdir, force = true, resume = false)

episode_count = 100
max_steps = 200
for i in 1:episode_count
  ob = env_reset(client, instance_id)
  done = false
  j = 1
  while j <= 200 && !done
    action = env_action_space_sample(client, instance_id)
    results = env_step(client, instance_id, action, render = true)
    done = results["done"]
    j = j + 1
  end
end

# Dump result info to disk
env_monitor_close(client, instance_id)
