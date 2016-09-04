/** Fields in these interfaces are named with underscores (e.g. `instance_id`)
 *  as opposed to in camel case (e.g. `instanceID`) because this is how they
 *  the server sends/expects them to be sent.
 */

// REPLY POST `/v1/envs/`
export interface NewEnvInstanceReply {
    instance_id: string;
}

// REPLY GET `/v1/envs/`
export interface GetEnvsReply {
    envs: { [envID: string]: string };
}

// REPLY POST `/v1/envs/<instance_id>/reset/`
export interface EnvResetReply {
    observation: any;
}

// REPLY POST `/v1/envs/<instance_id>/step/`
export interface StepReply {
    observation: any;
    reward: number;
    done: boolean;
    info: { [key: string]: any };
}

// REPLY GET `/v1/envs/<instance_id>/action_space/`
export interface ActionSpaceReply {
    info: { [name: string]: any };
}

// REPLY GET `/v1/envs/<instance_id>/observation_space/`
export interface ObservationSpaceReply {
    info: { [name: string]: any };
}
