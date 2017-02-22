package main

import (
	"fmt"

	gym "github.com/openai/gym-http-api/binding-go"
)

const BaseURL = "http://localhost:5000"

func main() {
	client, err := gym.NewClient(BaseURL)
	must(err)

	// Test the API for listing all instances.
	insts, err := client.ListAll()
	must(err)
	fmt.Println("Started with instances:", insts)

	// Create environment instance.
	id, err := client.Create("CartPole-v0")
	must(err)
	defer client.Close(id)

	// Test space information APIs.
	actSpace, err := client.ActionSpace(id)
	must(err)
	fmt.Printf("Action space: %+v\n", actSpace)
	obsSpace, err := client.ObservationSpace(id)
	must(err)
	fmt.Printf("Observation space: %+v\n", obsSpace)

	// Start monitoring to a temp directory.
	must(client.StartMonitor(id, "/tmp/cartpole-monitor", false, false, false))

	// Run through an episode.
	fmt.Println()
	fmt.Println("Starting new episode...")
	obs, err := client.Reset(id)
	must(err)
	fmt.Println("First observation:", obs)
	for {
		// Sample a random action to take.
		act, err := client.SampleAction(id)
		must(err)
		fmt.Println("Taking action:", act)

		// Unnecessary; demonstrates the ContainsAction API.
		c, err := client.ContainsAction(id, act)
		must(err)
		if !c {
			panic("sampled action not contained in space")
		}

		// Take the action, getting a new observation, a reward,
		// and a flag indicating if the episode is done.
		newObs, rew, done, _, err := client.Step(id, act, false)
		must(err)
		obs = newObs
		fmt.Println("reward:", rew, " -- observation:", obs)
		if done {
			break
		}
	}

	must(client.CloseMonitor(id))

	// Uncomment the code below to upload to the Gym website.
	// Note: you must set the OPENAI_GYM_API_KEY environment
	// variable or set the second argument of Upload() to a
	// non-empty string.
	//
	//     must(client.Upload("/tmp/cartpole-monitor", "", ""))
	//
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
