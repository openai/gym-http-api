import Foundation

let client = GymClient(baseURL: URL(string:"http://localhost:5000")!)
let existing = client.listAll()

let id = client.create(envID: "CartPole-v0")

print(client.actionSpace(instanceID: id))
print(client.observationSpace(instanceID: id))

client.startMonitor(instanceID: id, directory: "/tmp/swift-example-agent", force: true, resume: false, videoCallable: false)

let obs = client.reset(instanceID: id)

print("First observation: \(obs)")

var count = 0
while true {
    let action = client.sampleAction(instanceID: id)

    let result = client.step(instanceID: id, action: action)
    print("Result on iteration \(count). \nReward: \(result.reward). Observation: \(result.observation).")
    count += 1
    if result.done {
        break
    }
}

client.closeMonitor(instanceID: id)
client.close(instanceID: id)

// Get your api key from https://gym.openai.com/users/{your_name}
// client.uploadResults(directory: "/tmp/swift-example-agent", apiKey: nil, algorithmID: nil)

