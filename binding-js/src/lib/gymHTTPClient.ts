/// <reference path="../typings/index.d.ts" />

import * as ioInterfaces from "./ioInterfaces";
import * as axios from "axios";
import * as path from "path";

export default class Client {
    shouldLog: boolean;

    constructor(public remote) {
        this.shouldLog = (process.env.SHOULD_LOG === "true");
        if (this.shouldLog) {
            console.log("Logging enabled")
        } else {
            console.log("To enable logging, set SHOULD_LOG=true")
        }
    }

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
        return `${this.remote}${route}`;
    }

    _post<T>(route: string, data: any): Axios.IPromise<T> {
        if (this.shouldLog) {
            console.log(`POST ${route}\n${JSON.stringify(data)}`);
        }
        return this._parseServerErrors(axios.post(this._buildURL(route), data));
    }

    _get<T>(route: string): Axios.IPromise<T> {
        if (this.shouldLog) {
            console.log(`GET ${route}`);
        }
        return this._parseServerErrors(axios.get(this._buildURL(route)));
    }

    // POST `/v1/envs/`
    envCreate(envID: string): Axios.IPromise<ioInterfaces.NewEnvInstanceReply> {
        return this._post<ioInterfaces.NewEnvInstanceReply>("/v1/envs/",
            { env_id: envID }).then((value) => value);
    }

    // GET `/v1/envs/`
    envListAll(): Axios.IPromise<ioInterfaces.GetEnvsReply> {
        return this._get<ioInterfaces.GetEnvsReply>("/v1/envs/")
            .then((value) => value);
    }

    // POST `/v1/envs/<instanceID>/reset/`
    envReset(instanceID: string): Axios.IPromise<ioInterfaces.EnvResetReply> {
        const route = `/v1/envs/${instanceID}/reset/`;
        return this._post<ioInterfaces.EnvResetReply>(route, null)
            .then((value) => value.observation);
    }

    // POST `/v1/envs/<instanceID>/step/`
    envStep(instanceID: string, action: number, render: boolean = false):
        Axios.IPromise<ioInterfaces.StepReply> {
        const route = `/v1/envs/${instanceID}/step/`;
        const data = { action, render };
        return this._post<ioInterfaces.StepReply>(route, data)
            .then((value) => value);
    }

    // GET `/v1/envs/<instanceID>/action_space/`
    envActionSpaceInfo(instanceID: string):
        Axios.IPromise<ioInterfaces.ActionSpaceReply> {
        const route = `/v1/envs/${instanceID}/action_space/`
        return this._get<ioInterfaces.ActionSpaceReply>(route)
            .then((reply) => reply);
    }

    // GET `/v1/envs/<instanceID>/observation_space/`
    envObservationSpaceInfo(instanceID: string):
        Axios.IPromise<ioInterfaces.ObservationSpaceReply> {
        const route = `/v1/envs/${instanceID}/observation_space/`;
        return this._get<ioInterfaces.ObservationSpaceReply>(route)
            .then((reply) => reply);
    }

    // POST `/v1/envs/<instanceID>/monitor/start/`
    envMonitorStart(instanceID: string, directory: string,
        force: boolean = false, resume: boolean = false):
        Axios.IPromise<void> {
        const route = `/v1/envs/${instanceID}/monitor/start/`;
        return this._post(route, { directory, force, resume })
            .then((reply) => { return });
    }

    // POST `/v1/envs/<instanceID>/monitor/close/`
    envMonitorClose(instanceID: string): Axios.IPromise<void> {
        const route = `/v1/envs/${instanceID}/monitor/close/`
        return this._post(route, null)
            .then((reply) => { return });
    }

    // POST `/v1/envs/<instanceID>/close`
    envClose(instanceID: string): Axios.IPromise<void> {
        const route = `/v1/envs/${instanceID}/close/`
        return this._post(route, null)
            .then((reply) => { return });
    }

    // POST `/v1/upload/`
    upload(trainingDir: string, algorithmID: string = undefined, apiKey: string = undefined) {
        if (apiKey === undefined) {
            apiKey = process.env["OPENAI_GYM_API_KEY"];
        }
        this._post("/v1/upload/", {
            training_dir: trainingDir,
            algorithm_id: algorithmID,
            api_key: apiKey
        });
    }

    // POST `/v1/shutdown/`
    shutdownServer(self) {
        this._post("/v1/shutdown/", null)
    }
}
