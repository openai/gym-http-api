/// <reference path="../typings/index.d.ts" />

import * as ioInterfaces from "./ioInterfaces";
import * as axios from "axios";
import * as path from "path";

export class Client {
    constructor(public remoteBase: string, public port: number) { }

    _parseServerErrors<T>(promise: Axios.IPromise<Axios.AxiosXHR<T>>):
        Axios.IPromise<T> {
        return promise.then((response: Axios.AxiosXHR<T>) => {
            const { status } = response,
                message = response.data["message"];
            if (status !== 200 && message !== undefined) {
                throw new Error(`ERROR: Got status code ${status}\nMessage: ${message}`);
            } else {
                return response.data;
            }
        });
    }

    _buildURL(route: string): string {
        return `${this.remoteBase}/${route}`;
    }

    _post<T>(route: string, data: any): Axios.IPromise<Axios.AxiosXHR<T>> {
        console.log(`POST ${route}\n${JSON.stringify(data)}`);
        return this._parseServerErrors(axios.post(this._buildURL(route), data));
    }

    _get<T>(route: string): Axios.IPromise<Axios.AxiosXHR<T>> {
        console.log(`GET ${route}`);
        return this._parseServerErrors(axios.get(this._buildURL(route)));
    }

    env_create(env_id): Axios.IPromise<string> {
        return this._post<ioInterfaces.NewEnvInstanceReply>("/v1/envs/", 
            { env_id }).then((value) => value.data.instance_id);
    }

    env_list_all(): Axios.IPromise<ioInterfaces.GetEnvsReply> {
        return this._get<ioInterfaces.GetEnvsReply>("/v1/envs/")
            .then((value) => value.data);
    }

    env_reset(instance_id: string): Axios.IPromise<any> {
        const route = `/v1/envs/${instance_id}/reset`;
        return this._post<ioInterfaces.EnvResetReply>(route, null)
            .then((value) => value.data.observation);
    }

    env_step(instance_id: string, action: number, render: boolean = false):
        Axios.IPromise<ioInterfaces.StepReply> {
        const route = `/v1/envs/${instance_id}/step`;
        const data = { action, render };
        return this._post<ioInterfaces.StepReply>(route, data)
            .then((value) => value.data);
    }

    env_action_space_info(instance_id: string): Axios.IPromise<any> {
        const route = `/v1/envs/${instance_id}/action_space`
        return this._get<ioInterfaces.ActionSpaceReply>(route)
            .then((reply) => reply.data.info);
    }

    env_observation_space_info(instance_id: string): Axios.IPromise<any> {
        const route = `/v1/envs/${instance_id}/observation_space`;
        return this._get<ioInterfaces.ObservationSpaceReply>(route)
            .then((reply) => reply.data.info);
    }

    env_monitor_start(instance_id: string, directory: string,
        force: boolean = false, resume: boolean = false):
        Axios.IPromise<void> {
        const route = `/v1/envs/${instance_id}/monitor/start/`;
        return this._post(route, { directory, force, resume })
            .then((reply) => { return });
    }

    env_monitor_close(instance_id) {
        route = '/v1/envs/{}/monitor/close/'.format(instance_id)
        this._post(route, None)
    }

    env_close(instance_id) {
        route = '/v1/envs/{}/close/'.format(instance_id)
        this._post(route, None)
    }

    upload(training_dir, algorithm_id = None, api_key = None) {
        if not api_key:
            api_key = os.environ.get('OPENAI_GYM_API_KEY')

        this._post("/v1/upload/", {
            'training_dir': training_dir,
            'algorithm_id': algorithm_id,
            'api_key': api_key
        });
    }

    shutdown_server(self) {
        this._post('/v1/shutdown/', null)
    }
}
