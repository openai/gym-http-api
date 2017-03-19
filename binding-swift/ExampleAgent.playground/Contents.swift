import Foundation

let baseDirectory = "/tmp/swift-example-agent"

// Create a client
let client = GymClient(baseURL: URL(string:"http://localhost:5000")!)

// Make a new environment with the CartPole task
let id = client.create(envID: "CartPole-v0")

// Verify action and observation space
print(client.actionSpace(instanceID: id))
print(client.observationSpace(instanceID: id))
print(client.containsObservation(instanceID: id, observations: ["name":"Box"]))

// Start recording, and wipe out old recordings
client.startMonitor(instanceID: id, directory: baseDirectory, force: true, resume: false, videoCallable: true)

// Refresh to get our first observation
let obs = client.reset(instanceID: id)
print("First observation: \(obs)")

var count = 0
while true {
    // Find a random action purely as a demonstration. Replace with an action chosen by your algorithm.
    let action = client.sampleAction(instanceID: id)
    
    // Execute the action in the environment
    let result = client.step(instanceID: id, action: action)
    
    print("Result on iteration \(count). \nReward: \(result.reward). Observation: \(result.observation).")
    count += 1
    if result.done {
// Uncomment the following line to endlessly repeat
//        client.reset(instanceID: id)
        break
    }
}

// Clean up
client.closeMonitor(instanceID: id)
client.close(instanceID: id)

// Get your api key from https://gym.openai.com/users/{your_name}
// client.uploadResults(directory: baseDirectory, apiKey: nil, algorithmID: nil)

