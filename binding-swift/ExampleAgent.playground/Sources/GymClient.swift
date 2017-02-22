import Foundation

open class GymClient {
    let baseURL:URL
    
    public init(baseURL:URL) {
        self.baseURL = baseURL
    }
    
    open func listAll(callback:@escaping ([InstanceID:EnvID]) -> Void) {
        get(url: baseURL.appendingPathComponent("/v1/envs/")) { (json) in
            let map = (json as! [String:[InstanceID:EnvID]])["all_envs"]!
            callback(map)
        }
    }
    
    open func create(envID:EnvID, callback:@escaping (InstanceID) -> Void) {
        post(url: baseURL.appendingPathComponent("/v1/envs/"),
             parameter: ["env_id":envID]) { (json) in
                let instanceID = (json as! [String:InstanceID])["instance_id"]!
                callback(instanceID)
        }
    }
    
    
    // MARK: Helpers
    
    private func get(url:URL, parameter:Any? = nil, callback:@escaping (Any?) -> Void) {
        let task = URLSession.shared.dataTask(with: url) { (data, _, error) in
            if let error = error {
                print(error)
            }
            let json = try! JSONSerialization.jsonObject(with: data!, options: [])
            callback(json)
        }
        task.resume()
    }
    
    private func post(url:URL, parameter:Any, callback:@escaping (Any?) -> Void) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try! JSONSerialization.data(withJSONObject: parameter, options: [])
        let task = URLSession.shared.dataTask(with:request) { (data, _, error) in
            if let error = error {
                print(error)
            }
            let json = try! JSONSerialization.jsonObject(with: data!, options: [])
            callback(json)
        }
        task.resume()
    }
    
}

// MARK: Models

public struct Space {
    
    // Name is the name of the space, such as "Box", "HighLow",
    // or "Discrete".
    let name:String
    
    // Properties for Box spaces.
    let shape:[Int]?
    let low:[Double]?
    let high:[Double]?
    
    // Properties for Discrete spaces.
    let n:Int?
    
    // Properties for HighLow spaces.
    let numberOfRows:Int?
    let matrix:[Double]?
}

public typealias InstanceID = String
public typealias EnvID = String
