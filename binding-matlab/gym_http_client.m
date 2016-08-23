classdef gym_http_client < handle
    % Matlab Http client for OpenAI gym    
    
    properties
        remote_base        
    end
    
    methods (Access = 'private')
        % Parse a JSON response, and return a (struct) on resp_data
        % For more information about using JSON in matlab refer to:
        % http://mathworks.com/help/matlab/ref/webread.html
        % http://mathworks.com/help/matlab/ref/webwrite.html
        function [resp_data] = get_request(obj, route)            
            resp_data.observation = '';
        end
        
        % Encode a JSON message with data described on "req_data", and
        % return a response (struct) on resp_data
        function [resp_data] = post_request(obj, route, req_data)            
            url = [obj.remote_base, route];            
            options = weboptions('MediaType','application/json');
            resp_data = webwrite(url,req_data,options);
        end        
    end
    
    methods (Access = 'public')
        % Constructor
        function [objInstance] = gym_http_client(remote_base)
            objInstance.remote_base = remote_base;        
        end
        
        function [resp_data] = env_create(obj, env_id)
            route = '/v1/envs/';
            req_data = struct('env_id',env_id);
            resp_data = obj.post_request(route, req_data);
            resp_data = resp_data.instance_id;
        end
        
        function [resp_data] = env_list_all(obj)
            route = '/v1/envs/';
            resp_data = obj.get_request(obj, route);
            resp_data = resp_data.all_envs;
        end
        
        function [resp_data] = env_reset(obj, instance_id)
            route = ['/v1/envs/', instance_id, '/reset/'];            
            resp_data = obj.post_request(route, []);
            resp_data = resp_data.observation;
        end
        
        function [obs, reward, done, info] = env_step(obj, ...
                instance_id, action, render)            
            route = ['/v1/envs/', instance_id, '/step/'];
            req_data = struct('action',action,'render',render);
            resp_data = obj.post_request(route, req_data);
            obs = resp_data.observation;
            reward = resp_data.reward;
            done = resp_data.done;
            info = resp_data.info;
        end
        
        function [resp_data] = env_action_space_info(obj, instance_id)
            route = ['/v1/envs/', instance_id, '/action_space/'];            
            resp_data = obj.get_request(route);
            resp_data = resp_data.observation;
        end
        
        function [resp_data] = env_observation_space_info(obj, instance_id)
            route = ['/v1/envs/', instance_id, '/observation_space/'];            
            resp_data = obj.get_request(route);
            resp_data = resp_data.observation;
        end
        
        function [resp_data] = env_monitor_start(obj, ...
                instance_id, directory, varargin)
            if nargin > 3
                if nargin == 4
                    force = varargin{1};
                    resume = false;
                else
                    force = varargin{1};
                    resume = varargin{2};
                end
            else
                force = false;
                resume = false;
            end
            % Convert true/false to string
            if force
               force = 'true';
            else
                force = 'false';
            end
            if resume
               resume = 'true';
            else
                resume = 'false';
            end
            req_data = struct('directory',directory,'force',force,'resume',resume);            
            route = ['/v1/envs/', instance_id, '/monitor/start/'];  
            resp_data = obj.post_request(route, req_data);
        end
        
        function [resp_data] = env_monitor_close(obj, instance_id)            
            route = ['/v1/envs/', instance_id, '/monitor/close/'];  
            resp_data = obj.post_request(route, []);
        end
        
        function [resp_data] = upload(obj, ...
                training_dir, varargin)
            if nargin > 3                
                if nargin == 4
                    api_key = varargin{1};
                    algorithm_id = '';
                end
                if nargin == 5
                    api_key = varargin{1};
                    algorithm_id = varargin{2};
                end
            else
                api_key = getenv('OPENAI_GYM_API_KEY');
                algorithm_id = '';
            end            
            req_data = struct('training_dir',training_dir,...
                'algorithm_id',algorithm_id,'api_key',api_key);
            route = '/v1/upload/';
            resp_data = obj.post_request(route, req_data);
        end
       
        function shutdown_server(obj)
            route = '/v1/shutdown/';
            obj.post_request(route, []);
        end
    end    
end

