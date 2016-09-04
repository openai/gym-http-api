import { Client } from "../lib/gymHTTPClient"

class RandomDiscreteAgent {
    constructor(public n: number) { }

    act(observation, reward, done: boolean): number {
        return Math.floor(Math.random() * this.n);
    }
}

const client = new Client("127.0.0.1", 5000),
    envID = "CartPole-v0",
    numTrials = 3;

// Set up environment
let instanceID = client.envCreate(envID)

// Set up agent
const actionSpaceInfo = client.envActionSpaceInfo(instanceID)
const agent = new RandomDiscreteAgent(actionSpaceInfo.n)

// Run experiment, with monitor
const outdir = "/tmp/random-agent-results";
client.envMonitorStart(instanceID, outdir, true)

const episodeCount = 100
const maxSteps = 200
let reward = 0
let done = false

for (let i = 0; i < episodeCount; i++) {
    let ob = client.envReset(instanceID)
    for (let j = 0; j < maxSteps; j++) {
        let action = agent.act(ob, reward, done)
        let result = client.envStep(instanceID, action, true);
        ob = result.ob;
        reward = result.reward;
        done = result.done;
        if (done) {
            break
        }
    }
}

// Dump result info to disk
client.envMonitorClose(instanceID)

// Upload to the scoreboard. This expects the 'OPENAI_GYM_API_KEY'
// environment variable to be set on the client side.
console.log("Successfully ran example agent using " +
    "gym_http_client. Now trying to upload results to the " +
    "scoreboard. If this fails, you likely need to set " +
    "OPENAI_GYM_API_KEY=<your_api_key>");

client.upload(outdir)
