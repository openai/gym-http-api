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
    
    open func reset(instanceID:InstanceID, callback:@escaping (Observation) -> Void) {
        post(url: baseURL.appendingPathComponent("/v1/envs/reset/")) { (json) in
            let obs = (json as! [String:AnyObject])["observation"]!
            callback(Observation(base: obs))
        }
    }
    
    open func step(instanceID:InstanceID, action:Action, render:Bool = false, callback:@escaping (StepResult) -> Void) {
        let parameter = ["action":action.base, "render":render] as [String : Any]
        post(url: baseURL.appendingPathComponent("/\(instanceID)/step"), parameter: parameter) { (json) in
            let result = StepResult(jsonDict: json as! [String:AnyObject])
            callback(result)
        }
    }
    
    open func actionSpace(instanceID:InstanceID, callback:@escaping (Space) -> Void) {
        getSpace(instanceID: instanceID, name: "action_space", callback: callback)
    }
    
    open func observationSpace(instanceID:InstanceID, callback:@escaping (Space) -> Void) {
        getSpace(instanceID: instanceID, name: "observation_space", callback: callback)
    }
    
    open func sampleAction(instanceID:InstanceID, callback:@escaping (Action) -> Void) {
        get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/action_space/sample")) { (json) in
            let action = (json as! [String:AnyObject])["action"]!
            callback(Action(base:action))
        }
    }
    
    open func containsAction(instanceID:InstanceID, action:Action, callback:@escaping (Bool) -> Void) {
        guard action.discreteValue != nil else { fatalError("Currently only int action types are supported") }
        get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/action_space/contains")) { (json) in
            let member = (json as! [String:Bool])["member"]!
            callback(member)
        }
    }
    
    // MARK: Helpers
    
    private func get(url:URL, parameter:Any? = nil, callback:@escaping (Any?) -> Void) {
        let task = URLSession.shared.dataTask(with: url) { (data, res, error) in
            if let error = error {
                print(error)
            }
            
            let json = try! JSONSerialization.jsonObject(with: data!, options: [.allowFragments])
            callback(json)
        }
        task.resume()
    }
    
    private func post(url:URL, parameter:Any? = nil, callback:@escaping (Any?) -> Void) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        if let parameter = parameter {
            request.httpBody = try! JSONSerialization.data(withJSONObject: parameter, options: [])
        }
        let task = URLSession.shared.dataTask(with:request) { (data, _, error) in
            if let error = error {
                print(error)
            }
            let json = try! JSONSerialization.jsonObject(with: data!, options: [])
            callback(json)
        }
        task.resume()
    }
    
    private func getSpace(instanceID:InstanceID, name:String, callback:@escaping (Space) -> Void) {
        get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/\(name)/")) { (json) in
            let dict = (json as! [String:AnyObject])["info"] as! [String:AnyObject]
            let space = Space(jsonDict: dict)
            callback(space)
        }
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
    
    init(jsonDict:[String:AnyObject]) {
        name = jsonDict["name"] as! String
        
        shape = jsonDict["shape"] as! [Int]?
        low = jsonDict["low"] as! [Double]?
        high = jsonDict["high"] as! [Double]?
        n = jsonDict["n"] as! Int?
        numberOfRows = jsonDict["num_rows"] as! Int?
        matrix = jsonDict["matrix"] as! [Double]?
    }
}

public typealias InstanceID = String
public typealias EnvID = String

public typealias Action = MultiValueType
public typealias Observation = MultiValueType

public struct MultiValueType {
    let base:AnyObject
    
    init(base:AnyObject) {
        self.base = base
        if vectorValue == nil && discreteValue == nil {
            print("Unsupported value type: \(base)")
        }
    }
    
    var vectorValue:[Double]? {
        return base as? [Double]
    }
    
    var discreteValue:Int? {
        return base as? Int
    }
}

public struct StepResult {
    let observation:Observation
    let reward:Double
    let done:Bool
    let info:[String:AnyObject]
    
    init(jsonDict:[String:AnyObject]) {
        self.observation = Observation(base: jsonDict["observation"]!)
        self.reward =  jsonDict["reward"] as! Double
        self.done = jsonDict["done"] as! Bool
        self.info = jsonDict["info"] as! [String:AnyObject]
 
    }
}
