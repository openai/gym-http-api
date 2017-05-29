BeginPackage["GymEnvironment`", {"GeneralUtilities`"}]

EnvCreate;
EnvClose;
EnvListAll;
EnvReset;
EnvStep;

EnvActionSpaceInfo;
EnvActionSpaceSample;
EnvActionSpaceContains;
EnvObservationSpaceInfo;

EnvMonitorStart;
EnvMonitorClose;

ShutdownGymServer;

GymEnvironmentObject;

GymUpload;

Begin["`Private`"]

$DefaultServer = "http://127.0.0.1:5000";

(*----------------------------------------------------------------------------*)
(* Private functions *)

(*************************)
(* Factor out the commmon error handling routine *)

$unknownError = Failure["UnknownError", <|"MessageTemplate" :> "Error of unknown type."|>]; 

gymSafeRequestExecute[req_HTTPRequest] := Module[
	{res, body, msg},
	res = Quiet @ URLRead[req];
	If[FailureQ[res], Throw[res]];
	body = Quiet @ Developer`ReadRawJSONString[res["Body"]];
	(* sometimes, the body is a string that is not JSONizable, eg
		\"Server shutting down\". Need to handle this case.  *)
	msg = If[FailureQ[body], Missing[], Lookup[body, "message"]];
	If[(res["StatusCode"] =!= 200) && !MissingQ[msg],
		Throw @ Failure["ServerError", <|
			"MessageTemplate" :> StringTemplate["`Message`"], 
			"MessageParameters" -> <|"Message" -> msg|>
		|>]
	];
	body
]

gymSafeRequestExecute[_] := Throw[$unknownError]

(*************************)

gymPOSTRequest[server_String, route_String, data_Association] := gymSafeRequestExecute[
	HTTPRequest[server <> route, 
		<|
			"Body" -> Developer`WriteRawJSONString[data],
			Method -> "POST",
			"ContentType" -> "application/json"
		|>
	]
]

gymPOSTRequest[server_String, route_String] := gymPOSTRequest[server, route, <||>]


gymGETRequest[server_String, route_String] := gymSafeRequestExecute[
	HTTPRequest[server <> route, 
		<|
			Method -> "GET",
			"ContentType" -> "application/json"
		|>
	]
]

(*************************)
(* Make GymEnvironmentObject objects format nicely in notebooks *)

DefineCustomBoxes[GymEnvironmentObject, 
	e:GymEnvironmentObject[id_, name_, server_] :> Block[
	{},
	BoxForm`ArrangeSummaryBox[
		GymEnvironmentObject, e, 
		None, 
		{BoxForm`SummaryItem[{"ID: ", id}]
		 },
		{},
		StandardForm
	]]
];

(*----------------------------------------------------------------------------*)
(* Start of public API *)

(*************************)
SetUsage["
GymEnvironmentObject[id$]['ID'] returns the ID of the environment.
GymEnvironmentObject[id$]['Name'] returns the name of the environment.
GymEnvironmentObject[id$]['URL'] returns the URL of the server the environment \
is running on.
"]

GymEnvironmentObject[id_, _, _]["ID"] := id
GymEnvironmentObject[_, name_, _]["Name"] := name	
GymEnvironmentObject[_, _, url_]["URL"] := url	

(*************************)
SetUsage["
EnvListAll[] lists all environments running on the default server.
EnvListAll[url$] lists environments given the specified server url$, where url$ is \
either a string or a URL object.
"]

EnvListAll[server_String] := Catch @ Module[
	{res},
	res = gymGETRequest[server, "/v1/envs/"];
	res["all_envs"]
]

EnvListAll[URL[server]] := EnvListAll[server]
EnvListAll[] := EnvListAll[$DefaultServer];

(*************************)
SetUsage["
EnvCreate[type$] creates an instance of the environment with string \
name type$ on the default server.
EnvCreate[type$, url$] creates an environment on the specified server url$, \
where url$ is either a string or a URL object.
"]

EnvCreate[type_String, server_String] := Catch @ Module[
	{res},
	res = gymPOSTRequest[server, "/v1/envs/", <|"env_id" -> type|>];
	GymEnvironmentObject[res["instance_id"], type, server]
]

EnvCreate[type_String, URL[server_]] := 
	EnvCreate[type, server]

EnvCreate[type_String] :=
	EnvCreate[type, $DefaultServer]

(*************************)
SetUsage["
EnvClose[GymEnvironmentObject[id$]] closes the environment..
"]

EnvClose[ins_GymEnvironmentObject] := Catch @ Module[{},
	gymPOSTRequest[ins["URL"], StringJoin["/v1/envs/", ins["ID"], "/close/"]];
]

(*************************)	
SetUsage["
EnvStep[GymEnvironmentObject[id$], act$] steps through an environment using \
an action act$. EnvReset[GymEnvironmentObject[id$]] must be called before the first \
call to EnvStep. 
EnvStep[GymEnvironmentObject[id$], act$, render$] displays the current state \
of the environment in a separate windows when render$ is True.
"]

EnvStep[ins_GymEnvironmentObject, action_, render_:False] := Catch @ Module[
	{route, data},
	route = StringJoin["/v1/envs/", ins["ID"], "/step/"];	
	data = <|"action" -> action, "render" -> render|>;
	gymPOSTRequest[ins["URL"], route, data]
]

(*************************)	
SetUsage["
EnvReset[GymEnvironmentObject[id$]] resets the state of the environment and \
returns an initial observation.
"]
	
EnvReset[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/reset/"];
	res = gymPOSTRequest[ins["URL"], route];
	res["observation"]
]

(*************************)
SetUsage["
EnvActionSpaceInfo[GymEnvironmentObject[id$]] returns an Association \
containing information (name and dimensions/bounds) of the environment's action space.
"]

EnvActionSpaceInfo[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/action_space/"];
	res = gymGETRequest[ins["URL"], route];
	res["info"]
]

(*************************)
SetUsage["
EnvActionSpaceSample[GymEnvironmentObject[id$]] randomly samples an \
action from the action space.
"]

EnvActionSpaceSample[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/action_space/sample"];
	res = gymGETRequest[ins["URL"], route];
	res["action"]
]

(*************************)
SetUsage["
EnvActionSpaceContains[GymEnvironmentObject[id$], act$] returns True if act$ is \
an element of the action space, otherwise False.
"]

EnvActionSpaceContains[ins_GymEnvironmentObject, act_] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/action_space/contains/", ToString[act]];
	res = gymGETRequest[ins["URL"], route];
	res["member"]
]

(*************************)
SetUsage["
EnvObservationSpaceInfo[GymEnvironmentObject[id$]] gets information \
(name and dimensions/bounds) of the environments observation space.
"]

EnvObservationSpaceInfo[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/observation_space/"];
	res = gymGETRequest[ins["URL"], route];
	res["info"]
]

(*************************)
SetUsage["
EnvMonitorStart[GymEnvironmentObject[id$], dir$] starts logging actions and \
environment states to a file stored in dir$. The following options are available: 
| 'Force' | False | Clear out existing training data from this directory \
(by deleting every file prefixed with 'openaigym.') |
| 'Resume' | False | Retain the training data already in this directory, \
which will be merged with our new data. |
| 'VideoCallable' | False | Not yet implemented. |
"]

Options[EnvMonitorStart] =
{
	"Force" -> False,
	"Resume" -> False,
	"VideoCallable" -> False
};

EnvMonitorStart[ins_GymEnvironmentObject, dir_, opts:OptionsPattern[]] := Catch @ Module[
	{route, data},
	data = <|
		"directory" -> If[Head[dir] === File, First[dir], dir],
		"force" -> OptionValue["Force"],
		"resume" -> OptionValue["Resume"],
		"video_callable" -> OptionValue["VideoCallable"]
	|>;
	route = StringJoin["/v1/envs/", ins["ID"], "/monitor/start/"];	
	gymPOSTRequest[ins["URL"], route, data]; (* don't return *)
]

(*************************)
SetUsage["
EnvMonitorClose[GymEnvironmentObject[id$]] flushes all monitor data to disk.
"]

EnvMonitorClose[ins_GymEnvironmentObject] := Catch @ Module[{},
	gymPOSTRequest[ins["URL"], StringJoin["/v1/envs/", ins["ID"], "/monitor/close/"]];
]

(*************************)
SetUsage["
GymUpload[dir$] uploads results stored in dir$ to OpenAI Gym Server. Uses default \
server URL.
GymUpload[dir$, url$] uploads given a user-defined server url$.
| 'AlgorithmID' | '' | Name of algorithm |
| 'APIKey' | Automatic | When Automatic, tries to obtain key using GetEnvironment. \
otherwise, need to specify the key. |
"]

Options[GymUpload] =
{
	"AlgorithmID" -> "",
	"APIKey" -> Automatic
};

GymUpload[dir_String, url_String, opts:OptionsPattern[]] := Catch @ Module[
	{data, apiKey = OptionValue["APIKey"]},
	apiKey = If[apiKey === Automatic, 
		Values @ GetEnvironment["OPENAI_GYM_API_KEY"]
	];
	If[apiKey === None, 
		Throw @ Failure["GymAPIKey", <|"MessageTemplate" :> "No Gym API key defined."|>]
	];
	data = <|
		"training_dir" -> If[Head[dir] === File, First[dir], dir], 
		"algorithm_id" -> OptionValue["AlgorithmID"],
		"api_key" -> apiKey
	|>;
	gymPOSTRequest[url, "/v1/upload/", data]; (* don't return *)
]

GymUpload[dir_String, URL[url_], opts:OptionsPattern[]] := GymUpload[dir, url, opts]
GymUpload[dir_String, opts:OptionsPattern[]] := GymUpload[dir, $DefaultServer, opts]

(*************************)
SetUsage["
ShutdownGymServer[] requests a server shutdown at the default server URL.
ShutdownGymServer[url$] requests a shutdown of a server at the address url$, \
where url$ is either a string or URL object.
"]

ShutdownGymServer[url_String] := Catch @ Module[{},
	gymPOSTRequest[url, "/v1/shutdown/"];
]

ShutdownGymServer[URL[url_]] := ShutdownGymServer[url]
ShutdownGymServer[] := ShutdownGymServer[$DefaultServer]

(*----------------------------------------------------------------------------*)

End[ ]

EndPackage[ ]