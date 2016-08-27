export class Client {
    session: any;

    constructor(public remoteBase: string, public port: number) {
        this.session = requests.Session();
    }

    _parse_server_error_or_raise_for_status(resp) {
        let j = {}
        try {
            j = resp.json()
        } catch {
            // Most likely json parse failed because of network error, not server error (server
            // sends its errors in json). Don't let parse exception go up, but rather raise default
            // error.
            resp.raise_for_status()
        }
        if (resp.status_code != 200 and "message" in j) {  // descriptive message from server side
            raise ServerError(message=j["message"],
                              status_code=resp.status_code)
        }
        resp.raise_for_status()
        return j
    }

    _post_request(route, data) {
        url = urlparse.urljoin(this.remoteBase, route)
        logger.info("POST {}\n{}".format(url, json.dumps(data)))
        resp = this.session.post(urlparse.urljoin(this.remote_base, route),
                            data=json.dumps(data))
        return this._parse_server_error_or_raise_for_status(resp)
    }

    _get_request(route) {
        url = urlparse.urljoin(this.remote_base, route)
        logger.info("GET {}".format(url))
        resp = this.session.get(url)
        return this._parse_server_error_or_raise_for_status(resp)
    }
        
    env_create(env_id) {
        route = '/v1/envs/'
        data = {'env_id': env_id}
        resp = this._post_request(route, data)
        instance_id = resp['instance_id']
        return instance_id
    }

    env_list_all(self) {
        route = '/v1/envs/'
        resp = this._get_request(route)
        all_envs = resp['all_envs']
        return all_envs
    }

    env_reset(instance_id) {
        route = '/v1/envs/{}/reset/'.format(instance_id)
        resp = this._post_request(route, None)
        return resp['observation'];
    }

    env_step(instance_id, action, render=False) {
        route = '/v1/envs/{}/step/'.format(instance_id)
        data = {'action': action, 'render': render}
        resp = this._post_request(route, data)
        observation = resp['observation']
        reward = resp['reward']
        done = resp['done']
        info = resp['info']
        return [observation, reward, done, info]
    }

    env_action_space_info(instance_id) {
        route = '/v1/envs/{}/action_space/'.format(instance_id)
        resp = this._get_request(route)
        info = resp['info']
        return info
    }

    env_observation_space_info(instance_id) {
        route = '/v1/envs/{}/observation_space/'.format(instance_id)
        resp = this._get_request(route)
        info = resp['info']
        return info
    }

    env_monitor_start(instance_id, directory, force=False, resume=False) {
        route = '/v1/envs/{}/monitor/start/'.format(instance_id)
        data = {'directory': directory,
                'force': force,
                'resume': resume}
        this._post_request(route, data)
    }

    env_monitor_close(instance_id) {
        route = '/v1/envs/{}/monitor/close/'.format(instance_id)
        this._post_request(route, None)
    }

    env_close(instance_id) {
        route = '/v1/envs/{}/close/'.format(instance_id)
        this._post_request(route, None)
    }

    upload(training_dir, algorithm_id=None, api_key=None) {
        if not api_key:
            api_key = os.environ.get('OPENAI_GYM_API_KEY')

        route = '/v1/upload/'
        data = {'training_dir': training_dir,
                'algorithm_id': algorithm_id,
                'api_key': api_key}
        this._post_request(route, data)
    }

    shutdown_server(self) {
        this._post_request('/v1/shutdown/', null)
    }
}