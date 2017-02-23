
import PlaygroundSupport
import Foundation

//PlaygroundPage.current.needsIndefiniteExecution = true

let client = GymClient(baseURL: URL(string:"http://localhost:5000")!)
let existing = client.listAll()

let id:InstanceID = client.create(envID: "CartPole-v0")

client.startMonitor(instanceID: id, directory: "/tmp/swift-example-agent", force: true, resume: false, videoCallable: false)

let obs = client.reset(instanceID: id)

print("First observation: \(obs)")

while true {
    let action = client.sampleAction(instanceID: id)
    let result = client.step(instanceID: id, action: action)
    if result.done {
        print(result)
        break
    }
}

client.closeMonitor(instanceID: id)
client.uploadResults(directory: "/tmp/swift-example-agent", apiKey: nil, algorithmID: nil)

