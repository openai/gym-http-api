/// <reference path="../typings/index.d.ts" />

import Client from "../lib/gymHTTPClient"
import * as promise from "promise";

class RandomDiscreteAgent {
    constructor(public n: number) { }

    act(observation: any, reward: number, done: boolean): number {
        return Math.floor(Math.random() * this.n);
    }
}

const client = new Client("http://127.0.0.1:5000"),
    envID = "CartPole-v0",
    numTrials = 3,
    outDir = "/tmp/random-agent-results",
    episodeCount = 100,
    maxSteps = 200;

// Set asynchronously
let instanceID: string = undefined,
    agent: RandomDiscreteAgent = undefined;

/** Steps:
 *  1. Set up environment
 *  2. Set up agent
 *  3. Run experiment
 *  4. Try to upload results
 */
client.envCreate(envID)
    .then((reply) => {
        instanceID = reply.instance_id;
        return client.envActionSpaceInfo(instanceID)
    }).then((reply) => {
        agent = new RandomDiscreteAgent(reply.info["n"]);
        return client.envMonitorStart(instanceID, outDir, true);
    }).then(() => {
        return actOutEpisode(0)
            .then((done) => {
                client.envMonitorClose(instanceID).then((value) => { return; })
            });
    }).then(() => {
        console.log("Successfully ran example agent. Now trying to upload " +
            "results to the scoreboard. If this fails, you likely need to " +
            "set OPENAI_GYM_API_KEY=<your_api_key>");
        return client.upload(outDir)
    }).catch((error) => {
        console.log(`Example agent failed. Got error: ${error}`);
    });

// Recursively calls itself until `episode >= episodeCount`.
function actOutEpisode(episode: number): Promise.IThenable<boolean> {
    return new Promise((resolve, reject) => {
        if (episode >= episodeCount) {
            resolve(true);
        } else {
            client.envReset(instanceID)
                .then((reply) => actOutStep(0, 0, reply.observation, false))
                .then((done) => resolve(actOutEpisode(episode + 1)));
        }
    })
}

// Recursively calls itself until `step >= maxSteps`.
function actOutStep(step: number, reward: number, observation: any,
    done: boolean): Promise.IThenable<boolean> {
    return new Promise((resolve, reject) => {
        if (step >= maxSteps) {
            resolve(null);
        } else {
            let action = agent.act(observation, reward, done);
            client.envStep(instanceID, action, true)
                .then((reply) => {
                    if (reply.done) {
                        resolve(null);
                    } else {
                        resolve(actOutStep(step + 1, reply.reward,
                            reply.observation, reply.done));
                    }
                }).catch((error) => { throw error });
        }
    });
}
