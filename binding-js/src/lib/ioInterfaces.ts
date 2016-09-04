// REPLY GET `/v1/envs/`
export interface GetEnvsReply {
    envs: { [envID: string]: string };
}

// REPLY POST `/v1/envs/`
export interface NewEnvInstanceReply {
    instance_id: string;
}

// REPLY POST `/v1/envs/<instance_id>/step/`
export interface StepReply {
    observation: any;
    reward: number;
    done: boolean;
    info: any;
}

// REPLY POST `/v1/envs/<instance_id>/reset/`
export interface EnvResetReply {
    observation: any;
}

// REPLY GET `/v1/envs/<instance_id>/action_space/`
export interface ActionSpaceReply {
    info: { [name: string]: any };
}

// REPLY GET `/v1/envs/<instance_id>/observation_space/`
export interface ObservationSpaceReply {
    info: { [name: string]: any };
}