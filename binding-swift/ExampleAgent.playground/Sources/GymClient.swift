//
// GymClient.swift
// GYM-HTTP-API
// Created by Andrew Schreiber on 2/2/17.
//

// Add this file to your iOS/MacOS project to access a Gym HTTP server

import Foundation

open class GymClient {
    /// The URL for the Gym HTTP server
    public let baseURL:URL
    
    /// Creates an instance for interfacing with the Gym HTTP client.
    public init(baseURL:URL) {
        self.baseURL = baseURL
    }
    
    /// Create a new environment with a gym environment ID string i.e. "CartPole-v0"
    /// - returns : An instanceID to be used to uniquely identify the environment to be manipulated
    open func create(envID:EnvID) -> InstanceID {
        let json = post(url: baseURL.appendingPathComponent("/v1/envs/"),
                        parameter: ["env_id":envID])
        let instanceID = (json as! [String:InstanceID])["instance_id"]!
        return instanceID
    }
    
    /// Get all existing environment instances. The list is reset each time the server is reset.
    /// - returns : A dictionary of instanceIDs and the gym environment ID they are made from
    open func listAll() -> [InstanceID:EnvID] {
        let json = get(url: baseURL.appendingPathComponent("/v1/envs/"))
        let map = (json as! [String:[InstanceID:EnvID]])["all_envs"]!
        return map
    }
    
    /// Reset the state of the environment
    /// The resulting observation type may vary.
    /// For discrete spaces, it is an Int.
    /// For vector spaces, it is a [Double].
    /// - returns : An initial observation
    open func reset(instanceID:InstanceID) -> Observation {
        let json = post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/reset/"))
        let obs = (json as! [String:AnyObject])["observation"]!
        return Observation(base: obs)
    }
    
    /// Run one timestep of the environment's dynamics.
    /// - parameter action : An action to take. For discrete spaces, it should be an Int. For vector spaces, it should be a [Double].
    /// - parameter render : Undocumented functionality XD
    /// - returns : StepResult includes an observation of the current environment, amount of reward after the action, if the simulation is done, and any meta data
    
    open func step(instanceID:InstanceID, action:Action, render:Bool = false) -> StepResult {
        let parameter = ["action":action.base, "render":render] as [String : Any]
        let json = post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/step/"),
                        parameter: parameter)
        let result = StepResult(jsonDict: json as! [String:AnyObject])
        return result
    }
    
    /// Get information (name and dimensions/bounds) of the env's observation_space
    open func observationSpace(instanceID:InstanceID) -> Space {
        return getSpace(instanceID: instanceID, name: "observation_space")
    }
    
    
    /// Checks if observations are all contained in the observation space.
    open func containsObservation(instanceID:InstanceID, observations:[String:Any]) -> Bool {
        let json = post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/observation_space/contains"), parameter: observations)
        let isMember = (json as! [String:Bool])["member"]!
        return isMember
    }
    
    /// Get information (name and dimensions/bounds) of the env's action_space
    open func actionSpace(instanceID:InstanceID) -> Space {
        return getSpace(instanceID: instanceID, name: "action_space")
    }
    
    /// Sample an action randomly from all possible actions in the environment
    open func sampleAction(instanceID:InstanceID) -> Action {
        let json = get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/action_space/sample"))
        let action = (json as! [String:AnyObject])["action"]!
        return Action(base:action)
    }
    
    /// Checks if an action is contained in the action space. Currently, only int action types are supported
    open func containsAction(instanceID:InstanceID, action:Action) -> Bool {
        guard action.discreteValue != nil else { fatalError("Currently only int action types are supported") }
        let json = get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/action_space/contains/\(action.discreteValue!)"))
        let isMember = (json as! [String:Bool])["member"]!
        return isMember
    }
    
    /// Close the environment instance. Must be done before upload.
    open func close(instanceID:InstanceID) {
        _ = post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/close/"))
    }
    
    /// Start recording.
    /// - parameter directory : Location to write files. The server will create the directory if it does not exist.
    /// - parameter force : Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.")
    /// - parameter resume : Retain the training data already in this directory, which will be merged with our new data
    open func startMonitor(instanceID:InstanceID, directory:String, force:Bool, resume:Bool, videoCallable:Bool) {
    
        let parameter = ["directory":directory,
                         "force": force,
                         "resume":resume,
                         "video_callable": videoCallable] as [String : Any]
    
        _ = post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/monitor/start/"), parameter: parameter)
    }
    
    /// Stop recording and flush all data to disk.
    /// Two files will be created a meta data file like "openaigym.manifest.5.40273.manifest.json" and a performance file like "openaigym.episode_batch.11.40273.stats.json"
    open func closeMonitor(instanceID:InstanceID) {
       _ = post(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/monitor/close/"))
    }
    
    /// Shut down the server
    open func shutdown() {
        _ = post(url: baseURL.appendingPathComponent("/v1/shutdown/"))
    }
    
    /// Upload the results of training (as automatically recorded by your env's monitor) to OpenAI Gym.
    /// - parameter directory : Absolute path of directory containing recorder files i.e. "/tmp/swift-gym-agent"
    /// - parameter apiKey : Unique key from openai.com on your account page. Can be ignored if already contained in the environment.
    /// - parameter algorithmID : A unique identifer for the algorithm. You can safely leave this nil and it will be autogenerated.

    open func uploadResults(directory:String, apiKey:String?, algorithmID:String? = nil) {
        guard let apiKey = apiKey ?? environmentVariable(key:"OPENAI_GYM_API_KEY") else { fatalError("No API Key") }
        
        var data:[String:String] = ["training_dir": directory, "api_key": apiKey]
        if let algorithmID = algorithmID {
            data["algorithm_id"] = algorithmID
        }
        
        _ = post(url: baseURL.appendingPathComponent("/v1/upload/"), parameter: data)
    }
    
    // MARK: Helpers
    
    private func get(url:URL) -> Any? {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 120
        
        var json:Any?
        
        let semaphore = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: request) { (data, res, error) in
            self.httpErrorHandler(data: data, res: res, error: error)
            json = try! JSONSerialization.jsonObject(with: data!, options: [.allowFragments])
            semaphore.signal()
        }
        task.resume()
        _ = semaphore.wait(timeout: .distantFuture)
        return json
    }
    
    private func post(url:URL, parameter:Any? = nil) -> Any? {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 120
        
        if let parameter = parameter {
            request.httpBody = try! JSONSerialization.data(withJSONObject: parameter, options: [])
        }
        var json:Any?
        let semaphore = DispatchSemaphore(value: 0)

        let task = URLSession.shared.dataTask(with:request) { (data, res, error) in
            self.httpErrorHandler(data: data, res: res, error: error)
            
            json = try? JSONSerialization.jsonObject(with: data!, options: [.allowFragments])
            semaphore.signal()
            
        }
        task.resume()
        
        _ = semaphore.wait(timeout: .distantFuture)
        
        return json
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
    
    private func getSpace(instanceID:InstanceID, name:String) -> Space {
        let json = get(url: baseURL.appendingPathComponent("/v1/envs/\(instanceID)/\(name)/"))
        let dict = (json as! [String:AnyObject])["info"] as! [String:AnyObject]
        return Space(jsonDict: dict)
    }
}

// MARK: Models

public struct Space {
    
    // Name is the name of the space, such as "Box", "HighLow",
    // or "Discrete".
    public let name:String
    
    // Properties for Box spaces.
    public let shape:[Int]?
    public let low:[Double]?
    public let high:[Double]?
    
    // Properties for Discrete spaces.
    public let n:Int?
    
    // Properties for HighLow spaces.
    public let numberOfRows:Int?
    public let matrix:[Double]?
    
    public init(jsonDict:[String:AnyObject]) {
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
