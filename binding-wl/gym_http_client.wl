BeginPackage["GymEnvironment`", {"GeneralUtilities`"}]


EnvCreate;
EnvListAll;
EnvReset;
EnvStep;

EnvActionSpaceInfo;
EnvActionSpaceSample;
EnvActionSpaceContains;
EnvObservationSpaceInfo;

GymEnvironmentObject;

Begin["`Private`"]

$DefaultServer = "http://127.0.0.1:5000";

(*----------------------------------------------------------------------------*)

$unknownError = Failure["UnknownError", <|"MessageTemplate" :> "Error of unknown type."|>]; 

gymSafeRequestExecute[req_HTTPRequest] := Module[
	{res, body, msg},
	res = Quiet @ URLRead[req];
	If[FailureQ[res], Throw[res]];
	body = Quiet @ ImportString[res["Body"], "RawJSON"];
	If[!AssociationQ[body], Throw[$unknownError]];

	If[res["StatusCodeDescription"] =!= "OK", 
		msg = Lookup[body, "message", Throw[$unknownError]];
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
			"Body" -> ExportString[data, "RawJSON"],
			Method -> "POST",
			"ContentType" -> "application/json"
		|>
	]
]

gymGETRequest[server_String, route_String] := gymSafeRequestExecute[
	HTTPRequest[server <> route, 
		<|
			Method -> "GET",
			"ContentType" -> "application/json"
		|>
	]
]


(*************************)

EnvListAll[server_String] := Catch @ Module[
	{res},
	res = gymGETRequest[server, "/v1/envs/"];
	res["all_envs"]
]

EnvListAll[] := EnvListAll[$DefaultServer];

(*************************)

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

EnvStep[ins_GymEnvironmentObject, action_, render_:True] := Catch @ Module[
	{route, data},
	route = StringJoin["/v1/envs/", ins["ID"], "/step/"];	
	data = <|"action" -> action, "render" -> render|>;
	gymPOSTRequest[ins["URL"], route, data]
]

(*************************)	
	
EnvReset[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/reset/"];
	res = gymPOSTRequest[ins["URL"], route, <||>];
	res["observation"]
]

(*************************)	
	
EnvActionSpaceInfo[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/action_space/"];
	res = gymGETRequest[ins["URL"], route];
	res["info"]
]	

EnvActionSpaceSample[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/action_space/sample"];
	res = gymGETRequest[ins["URL"], route];
	res["action"]
]

EnvActionSpaceContains[ins_GymEnvironmentObject, x_] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/action_space/contains/", x];
	res = gymGETRequest[ins["URL"], route];
	res["member"]
]

EnvObservationSpaceInfo[ins_GymEnvironmentObject] := Catch @ Module[
	{route, res},
	route = StringJoin["/v1/envs/", ins["ID"], "/observation_space/"];
	res = gymGETRequest[ins["URL"], route];
	res["info"]
]
	
(*************************)

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

GymEnvironmentObject[id_, _, _]["ID"] := id
GymEnvironmentObject[_, name_, _]["Name"] := name	
GymEnvironmentObject[_, _, url_]["URL"] := url	
	
(*----------------------------------------------------------------------------*)

End[ ]

EndPackage[ ]