extern crate gym;

use gym::*;

#[test]
fn main() {
	println!("**********************************");

	let client = Client::new("http://localhost:5000".to_string());
	let mut env = client.make("CartPole-v0");

	//println!("observation space:\n{:?}", env.observation_space());
	//println!("action space:\n{:?}", env.action_space());

	for ep in 0..10 {
		let mut tot_reward = 0.;
		env.reset();
		for _ in 0.. {
			let action = env.action_space().sample();
			let state = env.step(action, true);
			assert_eq!(state.observation.len(), env.observation_space().sample().len());
			tot_reward += state.reward;

			if state.done {
				break;
			}
		}
		println!("Finished episode {} with total reward {}", ep, tot_reward);
	}

	println!("**********************************");
}