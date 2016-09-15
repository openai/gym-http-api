local HttpClient = require("httpclient")
local json = require("dkjson")
local os = require 'os'

local GymClient = {}
local m = {}

function m.new(remote_base)
   local self = {}
   self.remote_base = remote_base
   self.http = HttpClient.new()
   setmetatable(self, {__index = GymClient})
   return self
end

function GymClient:parse_server_error_or_raise_for_status(resp)
   local resp_data, pos, err = {}
   if resp.err then
      err = resp.err
      -- print('Response error: ' .. err)
   else
      if resp.code ~= 200 then
         err = resp.status_line
         -- Descriptive message from the server side
         -- print('Response: ' .. err)
      end
      resp_data, pos, err = json.decode(resp.body)
   end
   return resp_data, pos, err
end

function GymClient:get_request(route)
   url = self.remote_base .. route
   options = {}
   options.content_type = 'application/json'
   resp = self.http:get(url, options)
   return self:parse_server_error_or_raise_for_status(resp)
end

function GymClient:post_request(route, req_data)
   url = self.remote_base .. route
   options = {}
   options.content_type = 'application/json'
   json_str = json.encode(req_data)
   resp = self.http:post(url, json_str, options)
   return self:parse_server_error_or_raise_for_status(resp)
end

function GymClient:env_create(env_id)
   route = '/v1/envs/'
   req_data = {env_id = env_id}
   resp_data = self:post_request(route, req_data)
   return resp_data['instance_id']
end

function GymClient:env_list_all()
   route = '/v1/envs/'
   resp_data = self:get_request(route)
   return resp_data['all_envs']
end

function GymClient:env_reset(instance_id)
   route = '/v1/envs/'..instance_id..'/reset/'
   resp_data = self:post_request(route, '')
   return resp_data['observation']
end

function GymClient:env_step(instance_id, action, render, video_callable)
   render = render or false
   video_callable = video_callable or false
   route = '/v1/envs/'..instance_id..'/step/'
   req_data = {action = action, render = render, video_callable = video_callable}
   resp_data = self:post_request(route, req_data)
   obs = resp_data['observation']
   reward = resp_data['reward']
   done = resp_data['done']
   info = resp_data['info']
   return obs, reward, done, info
end

function GymClient:env_action_space_info(instance_id)
   route = '/v1/envs/'..instance_id..'/action_space/'
   resp_data = self:get_request(route)
   return resp_data['info']
end

function GymClient:env_action_space_sample(instance_id)
   route = '/v1/envs/'..instance_id..'/action_space/sample'
   resp_data = self:get_request(route)
   action = resp_data['action']
   return action
end

function GymClient:env_action_space_contains(instance_id)
   route = '/v1/envs/'..instance_id..'/action_space/contains'
   resp_data = self:get_request(route)
   member = resp['member']
   return member
end

function GymClient:env_observation_space_info(instance_id)
   route = '/v1/envs/'..instance_id..'/observation_space/'
   resp_data = self:get_request(route)
   return resp_data['info']
end

function GymClient:env_monitor_start(instance_id, directory, force, resume, video_callable)
   if not force then force = false end
   if not resume then resume = false end
   req_data = {directory = directory,
            force = tostring(force),
            resume = tostring(resume),
            video_callable = video_callable}
   route = '/v1/envs/'..instance_id..'/monitor/start/'
   resp_data  = self:post_request(route, req_data)
end

function GymClient:env_monitor_close(instance_id)
   route = '/v1/envs/'..instance_id..'/monitor/close/'
   resp_data = self:post_request(route, '')
end

function GymClient:upload(training_dir, algorithm_id, api_key)
   if not api_key then
     api_key = os.getenv('OPENAI_GYM_API_KEY')
   end
   if not algorithm_id then algorithm_id = '' end
   req_data = {training_dir = training_dir,
            algorithm_id = algorithm_id,
            api_key = api_key}
   route = '/v1/upload/'
   resp = self:post_request(route, req_data)
   return resp
end

function GymClient:shutdown_server()
   route = '/v1/shutdown/'
   self:post_request(route, '')
end

return m