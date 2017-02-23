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
        post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/reset/")) { (json) in
            let obs = (json as! [String:AnyObject])["observation"]!
            callback(Observation(base: obs))
        }
    }
    
    open func step(instanceID:InstanceID, action:Action, render:Bool = false, callback:@escaping (StepResult) -> Void) {
        let parameter = ["action":action.base, "render":render] as [String : Any]
        post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/step/"), parameter: parameter) { (json) in
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
    
    private func getSpace(instanceID:InstanceID, name:String, callback:@escaping (Space) -> Void) {
        get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/\(name)/")) { (json) in
            let dict = (json as! [String:AnyObject])["info"] as! [String:AnyObject]
            let space = Space(jsonDict: dict)
            callback(space)
        }
    }
    open func sampleAction(instanceID:InstanceID, callback:@escaping (Action) -> Void) {
        get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/action_space/sample")) { (json) in
            let action = (json as! [String:AnyObject])["action"]!
            callback(Action(base:action))
        }
    }
    
    open func containsAction(instanceID:InstanceID, action:Action, callback:@escaping (Bool) -> Void) {
        guard action.discreteValue != nil else { fatalError("Currently only int action types are supported") }
        get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/action_space/contains/")) { (json) in
            let member = (json as! [String:Bool])["member"]!
            callback(member)
        }
    }
    
    open func close(instanceID:InstanceID) {
        post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/close/")) { (json) in
            print("Result of close: \(json)")
        }
    }
    
    open func startMonitor(instanceID:InstanceID, directory:String, force:Bool, resume:Bool, videoCallable:Bool, callback:@escaping (Void) -> Void) {
    
        let parameter = ["directory":directory,
                         "force": force,
                         "resume":resume,
                         "video_callable": videoCallable] as [String : Any]
    
        post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/monitor/start/"), parameter: parameter) { (json) in
            print("Result of start monitor \(json)")
            callback()
        }
    }
    
    open func closeMonitor(instanceID:InstanceID) {
        post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/monitor/close/")) { (json) in
            print("Result of close monitor \(json)")
        }
    }
    
    open func shutdown() {
        post(url: baseURL.appendingPathComponent("/v1/shutdown/")) { (json) in
            print("Result of shutdown \(json)")
        }
    }
    
    open func uploadResults(directory:String, apiKey:String?, algorithmID:String?) {
        guard let apiKey = apiKey ?? environmentVariable(key:"OPENAI_GYM_API_KEY") else { fatalError("No API Key") }
        
        var data:[String:String] = ["training_dir": directory, "api_key": apiKey]
        if let algorithmID = algorithmID {
            data["algorithm_id"] = algorithmID
        }
        
        post(url: baseURL.appendingPathComponent("/v1/upload/"), parameter: data) { (json) in
            print("Result of upload \(json)")
        }
    }
    
    // MARK: Helpers
    
    private func get(url:URL, parameter:Any? = nil, callback:@escaping (Any?) -> Void) {
        let task = URLSession.shared.dataTask(with: url) { (data, res, error) in
            self.httpErrorHandler(data: data, res: res, error: error)
            
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
        let task = URLSession.shared.dataTask(with:request) { (data, res, error) in
            self.httpErrorHandler(data: data, res: res, error: error)
            
            let json = try? JSONSerialization.jsonObject(with: data!, options: [.allowFragments])
            callback(json ?? [:])
        }
        task.resume()
    }
    
    private func httpErrorHandler(data:Data?, res:URLResponse?, error:Error?) {
        if let error = error {
            fatalError(error.localizedDescription)
        } else if let res = res as? HTTPURLResponse, ![200, 204].contains(res.statusCode) {
            let text = String(data:data!, encoding:String.Encoding.utf8)!
            fatalError("Error with request:\(text). Response: \(res)")
        }
    }
    private func environmentVariable(key:String) -> String? {
        guard let rawValue = getenv(key) else { return nil }
        return String(utf8String: rawValue)
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
    public let base:AnyObject
    
    public init(base:AnyObject) {
        self.base = base
        if vectorValue == nil && discreteValue == nil {
            print("Unsupported value type: \(base)")
        }
    }
    
    public var vectorValue:[Double]? {
        return base as? [Double]
    }
    
    public var discreteValue:Int? {
        return base as? Int
    }
}

public struct StepResult {
    public let observation:Observation
    public let reward:Double
    public let done:Bool
    public let info:[String:AnyObject]
    
    public init(jsonDict:[String:AnyObject]) {
        self.observation = Observation(base: jsonDict["observation"]!)
        self.reward =  jsonDict["reward"] as! Double
        self.done = jsonDict["done"] as! Bool
        self.info = jsonDict["info"] as! [String:AnyObject]
 
    }
}
