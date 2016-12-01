# Start python server


require_relative "gym_client.rb"

remote_base = 'http://127.0.0.1:5000'
client = Openai::Client.new(remote_base)

# Create environment
env_id = 'MountainCar-v0'
instance_id = client.env_create(env_id)

# Check properties
all_envs = client.env_list_all
action_info = client.env_action_space_info(instance_id)

obs_info = client.env_observation_space_info(instance_id)

# Run a single step
client.env_monitor_start(instance_id, directory='tmp', force=true)
init_obs = client.env_reset(instance_id)
arr = client.env_step(instance_id, 1, true)
client.env_monitor_close(instance_id)

# API Key required
client.upload(training_dir='tmp')
