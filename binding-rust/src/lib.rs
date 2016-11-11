extern crate serde_json;
extern crate hyper;
extern crate rand;

use std::collections::BTreeMap;
use std::io::Read;

use serde_json::Value;
use serde_json::value::{ToJson, from_value};

use hyper::client::Client;
use hyper::header::Headers;

use rand::{thread_rng, Rng};

pub type GymResult<T> = Result<T, hyper::Error>;

#[derive(Debug, Clone)]
pub enum Space {
	DISCRETE{n: u64},
	BOX{shape: Vec<u64>, high: Vec<f64>, low: Vec<f64>},
	TUPLE{spaces: Vec<Box<Space>>}
}

impl Space {
	fn from_json(info: &Value) -> Space {
		match info.find("name").unwrap().as_str().unwrap() {
			"Discrete" => {
				let n = info.find("n").unwrap().as_u64().unwrap();
				Space::DISCRETE{n: n}
			},
			"Box" => {
				let shape = info.find("shape").unwrap().as_array().unwrap()
								.into_iter().map(|x| x.as_u64().unwrap())
								.collect::<Vec<_>>();


				let high = info.find("high").unwrap().as_array().unwrap()
							   .into_iter().map(|x| x.as_f64().unwrap())
							   .collect::<Vec<_>>();
							   
				let low = info.find("low").unwrap().as_array().unwrap()
							  .into_iter().map(|x| x.as_f64().unwrap())
							  .collect::<Vec<_>>();

				Space::BOX{shape: shape, high: high, low: low}
			},
			"Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
			e @ _ => panic!("Unrecognized space name: {}", e)
		}
	}
	pub fn sample(&self) -> Vec<f64> {
		let mut rng = thread_rng();
		match *self {
			Space::DISCRETE{n} => {
				vec![(rng.gen::<u64>()%n) as f64]
			},
			Space::BOX{ref shape, ref high, ref low} => {
				let mut ret = Vec::with_capacity(shape.iter().map(|x| *x as usize).product());
				let mut index = 0;
				for &i in shape {
					for _ in 0..i {
						ret.push(rng.gen_range(low[index], high[index]));
						index += 1;
					}
				}
				ret
			},
			Space::TUPLE{ref spaces} => {
				let mut ret = Vec::new();
				for space in spaces {
					ret.extend(space.sample());
				}
				ret
			}
		}
	}
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct State {
	pub observation:	Vec<f64>,
	pub reward:			f64,
	pub done:			bool,
	pub info:			Value,
}

#[allow(dead_code)]
pub struct Environment {
	client:			GymClient,
	instance_id:	String,
	act_space:		Space,
	obs_space:		Space,
}

impl Environment {
	pub fn action_space<'a>(&'a self) -> &'a Space {
		&self.act_space
	}
	pub fn observation_space<'a>(&'a self) -> &'a Space {
		&self.obs_space
	}
	pub fn reset(&mut self) -> GymResult<Vec<f64>> {
		let path = "/v1/envs/".to_string() + &self.instance_id + "/reset/";
		let observation = try!(self.client.post(path, Value::Null));

		let ret: Vec<_> = observation.find("observation").unwrap().as_array().unwrap()
									 .into_iter().map(|x| x.as_f64().unwrap())
									 .collect();
		Ok(ret)
	}
	pub fn step(&mut self, action: Vec<f64>, render: bool) -> GymResult<State> {
		let mut req = BTreeMap::new();
		req.insert("render", Value::Bool(render));
		match self.act_space {
			Space::DISCRETE{..} => {
				assert_eq!(action.len(), 1);
				req.insert("action", Value::U64(action[0] as u64));
			},
			Space::BOX{ref shape, ..} => {
				assert_eq!(action.len(), shape[0] as usize);
				req.insert("action", action.to_json());
			},
			Space::TUPLE{..} => panic!("Actions for Tuple spaces not implemented yet")
		}
		
		let path = "/v1/envs/".to_string() + &self.instance_id + "/step/";
		let state = try!(self.client.post(path, req.to_json()));

		Ok(State {
			observation: from_value(state.find("observation").unwrap().clone()).unwrap(),
			reward: state.find("reward").unwrap().as_f64().unwrap(),
			done: state.find("done").unwrap().as_bool().unwrap(),
			info: state.find("info").unwrap().clone()
		})
	}
	pub fn monitor_start(&mut self, directory: String, force: bool, resume: bool) -> GymResult<()> {
		let mut req = BTreeMap::new();
		req.insert("directory", Value::String(directory));
		req.insert("force", Value::Bool(force));
		req.insert("resume", Value::Bool(resume));

		let path = "/v1/envs/".to_string() + &self.instance_id + "/monitor/start/";
		try!(self.client.post(path, req.to_json()));
		Ok(())
	}
	pub fn monitor_stop(&mut self) -> GymResult<()> {
		let path = "/v1/envs/".to_string() + &self.instance_id + "/monitor/close/";
		try!(self.client.post(path, Value::Null));
		Ok(())
	}
	pub fn upload(&mut self, training_dir: String, api_key: String, algorithm_id: String) -> GymResult<()> {
		let mut req = BTreeMap::new();
		req.insert("training_dir", training_dir);
		req.insert("api_key", api_key);
		req.insert("algorithm_id", algorithm_id);

		try!(self.client.post("/v1/upload/".to_string(), req.to_json()));
		Ok(())
	}
}

pub struct GymClient {
	address:	String,
	handle:		Client,
	headers:	Headers,
}

impl GymClient {
    pub fn new(addr: String) -> GymClient {
		let mut headers = Headers::new();
		headers.set_raw("Content-Type", vec![b"application/json".to_vec()]);

    	GymClient {
    		address: addr, 
    		handle: Client::new(),
    		headers: headers
    	}
    }
    pub fn make(mut self, env_id: &str) -> GymResult<Environment> {
    	let mut req: BTreeMap<&str, &str> = BTreeMap::new();
    	req.insert("env_id", env_id);

    	let instance_id = try!(self.post("/v1/envs/".to_string(), req.to_json()));
    	let instance_id = match instance_id.find("instance_id") {
    		Some(id) => id.as_str().unwrap(),
    		None => panic!("Unrecognized environment id: {}", env_id)
    	};

    	let obs_space = try!(self.get("/v1/envs/".to_string() + instance_id + "/observation_space/"));
    	let act_space = try!(self.get("/v1/envs/".to_string() + instance_id + "/action_space/"));

    	Ok(Environment {
    		client: self,
    		instance_id: instance_id.to_string(),
    		act_space: Space::from_json(act_space.find("info").unwrap()),
    		obs_space: Space::from_json(obs_space.find("info").unwrap())})
    }
    pub fn get_envs(&mut self) -> GymResult<BTreeMap<String, String>> {
    	let json = try!(self.get("/v1/envs/".to_string()));
    	Ok(from_value(json.find("all_envs").unwrap().clone()).unwrap())
    }

    fn post(&mut self, route: String, request: Value) -> GymResult<Value> {
    	let url = self.address.clone() + &route;
    	let mut resp = try!(self.handle.post(&url)
    							  	   .body(&request.to_string())
    							  	   .headers(self.headers.clone())
    							  	   .send());

    	let mut json = String::new();
    	let _ = resp.read_to_string(&mut json);

	    Ok(serde_json::from_str(&json).unwrap_or(Value::Null))
    }
    fn get(&mut self, route: String) -> GymResult<Value> {
    	let url = self.address.clone() + &route;
    	let mut resp = try!(self.handle.get(&url)
    							  	   .send());
    	let mut json = String::new();
    	let _ = resp.read_to_string(&mut json);

		Ok(serde_json::from_str(&json).unwrap_or(Value::Null))
    }
}