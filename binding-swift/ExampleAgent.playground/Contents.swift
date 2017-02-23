
import PlaygroundSupport
import Foundation

PlaygroundPage.current.needsIndefiniteExecution = true

let client = GymClient(baseURL: URL(string:"http://localhost:5000")!)

//client.listAll { instanceEnvMap in
////    print("Got map: \(instanceEnvMap)")
//    if let instanceID = instanceEnvMap.first?.key{
////        client.startMonitor(instanceID: instanceID, directory: "/tmp/cartpole-monitor", force: false, resume: false, videoCallable: false)
//        
////        client.close(instanceID: instanceID)
////        client.closeMonitor(instanceID: instanceID)
//        client.actionSpace(instanceID: instanceID, callback: { (space) in
//            print("Got space succeeded")
//        })
//        client.actionSpace(instanceID: instanceID, callback: { (space) in
//            print("Got space succeeded")
//        })
////        client.step(instanceID: instanceID, action: Action(base:[5.0] as AnyObject), callback: { (step) in
////            print("Got step result \(step)")
////        })
//    }
//}

client.create(envID: "CartPole-v0") { (instanceID) in
    print("Create succeeded: \(instanceID)")
//    defer { client.close(instanceID: instanceID) }
//
//    print("Got instance ID: \(instanceID)")
//    
    client.observationSpace(instanceID: instanceID) { (space) in
        print(space)
    }
    client.actionSpace(instanceID: instanceID) { (space) in
        print(space)
    }
    client.startMonitor(instanceID: instanceID, directory: "/tmp/cartpole-monitor", force: true, resume: false, videoCallable: false) {
        client.reset(instanceID: instanceID) { (obs) in
            var stepping = true
            var count = 0
    
            print("Starting repeat")
            repeat {
                count += 1
                client.sampleAction(instanceID: instanceID) { (action) in
                    print("got action: \(action)")

                    
                    client.step(instanceID: instanceID, action: action) { (result) in
                        print(result)
                        if result.done {
                            stepping = false
                        }
                    }
                }
                
            } while(stepping && count < 10000)
            
        }
    }
}
