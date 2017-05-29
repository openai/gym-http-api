$Package = FileNameJoin[{DirectoryName @ $InputFileName, "gym_http_client.wl"}]
Get[$Package]


(* This script will run an agent with random actions *)

env = EnvCreate["CartPole-v0"]

$numEpisodes = 100;
$maxSteps = 200;

Do[
	EnvReset[env]; (* reset every episode *)
	Do[
		
  		action = EnvActionSpaceSample[env];
  		state = EnvStep[env, action, True];
  		If[state["done"], Break[]],
  		{step, $maxSteps}
  	],
  	{episode, $numEpisodes}
 ]
 
 (* close the environment when done *)
 EnvClose[env]