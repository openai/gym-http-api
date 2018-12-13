extern crate serde;
extern crate serde_json;

extern crate reqwest;

extern crate rand;

mod error;
mod space;

use std::collections::BTreeMap;

use serde::ser::Serialize;
use serde_json::Value;
use serde_json::value::from_value;

use reqwest::Client;
use reqwest::header::ContentType;

pub use self::error::{GymResult, GymError};
pub use self::space::Space;

#[derive(Debug)]
pub struct State {
    pub observation:    Vec<f64>,
    pub reward:         f64,
    pub done:           bool,
    pub info:           Value,
}

pub struct Environment {
    client:         GymClient,
    instance_id:    String,
    act_space:      Space,
    obs_space:      Space,
}

impl Environment {
    pub fn action_space<'a>(&'a self) -> &'a Space {
        &self.act_space
    }
    pub fn observation_space<'a>(&'a self) -> &'a Space {
        &self.obs_space
    }
    pub fn reset(&self) -> GymResult<Vec<f64>> {
        let path = "/v1/envs/".to_string() + &self.instance_id + "/reset/";
        let observation = self.client.post(path, &Value::Null)?;

        Ok(from_value(observation["observation"].clone())
            .expect("Should only panic if the API changes"))
    }
    pub fn step(&self, action: Vec<f64>, render: bool) -> GymResult<State> {
        let mut req = BTreeMap::new();
        req.insert("render", Value::Bool(render));
        match self.act_space {
            Space::DISCRETE{..} => {
                debug_assert_eq!(action.len(), 1);
                req.insert("action", (action[0] as u64).into());
            },
            Space::BOX{ref shape, ..} => {
                debug_assert_eq!(action.len(), shape.iter().map(|&x| x as usize).product::<usize>());
                req.insert("action", action.into());
            },
            Space::TUPLE{..} => panic!("Actions for Tuple spaces not implemented yet")
        }
        
        let path = "/v1/envs/".to_string() + &self.instance_id + "/step/";
        let state = self.client.post(path, &req)?;

        Ok(State {
            observation: from_value(state["observation"].clone()).unwrap(),
            reward: state["reward"].as_f64().unwrap(),
            done: state["done"].as_bool().unwrap(),
            info: state["info"].clone()
        })
    }
    pub fn monitor_start(&self, directory: String, force: bool, resume: bool) -> GymResult<Value> {
        let mut req = BTreeMap::new();
        req.insert("directory", Value::String(directory));
        req.insert("force", Value::Bool(force));
        req.insert("resume", Value::Bool(resume));

        let path = "/v1/envs/".to_string() + &self.instance_id + "/monitor/start/";
        self.client.post(path, &req)
    }
    pub fn monitor_stop(&self) -> GymResult<Value> {
        let path = "/v1/envs/".to_string() + &self.instance_id + "/monitor/close/";
        self.client.post(path, &Value::Null)
    }
    pub fn upload(&self, training_dir: String, api_key: String, algorithm_id: String) -> GymResult<Value> {
        let mut req = BTreeMap::new();
        req.insert("training_dir", training_dir);
        req.insert("api_key", api_key);
        req.insert("algorithm_id", algorithm_id);

        self.client.post("/v1/upload/".to_string(), &req)
    }
}

pub struct GymClient {
    address:    String,
    handle:     Client,
}

impl GymClient {
    pub fn new(addr: String) -> GymResult<GymClient> {
        Ok(GymClient {
            address: addr,
            handle: Client::new()?,
        })
    }
    pub fn make(self, env_id: &str) -> GymResult<Environment> {
        let mut req: BTreeMap<&str, &str> = BTreeMap::new();
        req.insert("env_id", env_id);

        let instance_id = self.post("/v1/envs/".to_string(), &req)?;
        let instance_id = instance_id["instance_id"].as_str().unwrap();

        let obs_space = self.get("/v1/envs/".to_string() + instance_id + "/observation_space/")?;
        let act_space = self.get("/v1/envs/".to_string() + instance_id + "/action_space/")?;

        Ok(Environment {
            client: self,
            instance_id: instance_id.to_string(),
            act_space: Space::from_json(&act_space["info"])?,
            obs_space: Space::from_json(&obs_space["info"])?})
    }
    pub fn get_envs(&self) -> GymResult<BTreeMap<String, String>> {
        let json = self.get("/v1/envs/".to_string())?;
        Ok(from_value(json["all_envs"].clone()).unwrap())
    }


    fn post<T: Serialize>(&self, route: String, request: &T) -> GymResult<Value> {
        let url = self.address.clone() + &route;
        match self.handle.post(&url)?.header(ContentType::json()).json(request)?.send()?.json() {
            Ok(val) => Ok(val),
            Err(e)  => Err(e.into()),
        }
     
    }
    fn get(&self, route: String) -> GymResult<Value> {
        let url = self.address.clone() + &route;
        match self.handle.get(&url)?.send()?.json() {
            Ok(val) => Ok(val),
            Err(e)  => Err(e.into()),
        }
    }
}