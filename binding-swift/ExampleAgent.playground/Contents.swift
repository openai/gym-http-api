
import PlaygroundSupport
import Foundation

PlaygroundPage.current.needsIndefiniteExecution = true

let client = GymClient(baseURL: URL(string:"http://localhost:5000")!)

client.listAll { instanceEnvMap in
    print("Got map: \(instanceEnvMap)")
}

client.create(envID: "CartPole-v0") { (instanceID) in
    print("Got instance ID: \(instanceID)")
}



