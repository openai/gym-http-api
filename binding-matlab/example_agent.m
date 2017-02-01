%% Example using matlab gym_http_client interface
% Example:
% 1) Server: python gym_http_server.py
% 2) Client: matlab -nojvm -nodisplay -nosplash -r "run example_agent.m"
fprintf('Testing Matlab client\n');

%% Setup client
base = 'http://127.0.0.1:5000';
client = gym_http_client(base);

%% Set up enviroment
env_id = 'CartPole-v0';
instance_id = client.env_create(env_id);

%% Run random experiment with monitor
outdir = '/tmp/random-matlab-agent-results';
client.env_monitor_start(instance_id, outdir, true);
render = false;

episode_count = 100;
max_steps = 200;
reward = 0;
done = false;

for i = 1:episode_count
   obs = client.env_reset(instance_id);
   for j=1:max_steps
       action = client.env_action_space_sample(instance_id);
       [ob, reward, done, info] = ...
           client.env_step(instance_id, action, render);
       if done
          break;
       end
   end
end

%% Dump result info to disk
client.env_monitor_close(instance_id);

%% Upload to the scoreboard.
% This expects the 'OPENAI_GYM_API_KEY' enviroment variable to be set on
% the client side.
client.upload(outdir);

fprintf('Matlab client test ended\n');
