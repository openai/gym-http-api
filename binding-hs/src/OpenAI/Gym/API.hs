-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.API
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
module OpenAI.Gym.API where

import OpenAI.Gym.Prelude
import OpenAI.Gym.Data

-- | a description of the Gym HTTP API
type GymAPI
  = "v1" :> ( "envs" :> ( ReqBody '[JSON] GymEnv :> Post '[JSON] InstID
                     :<|> Get '[JSON] Environment
                     :<|> Capture "instance_id" InstID :> "reset" :> Post '[JSON] Observation
                     :<|> Capture "instance_id" InstID :> "step"  :> ReqBody '[JSON] Step :> Post '[JSON] Outcome
                     :<|> Capture "instance_id" InstID :> "action_space" :> Get '[JSON] Info
                     :<|> Capture "instance_id" InstID :> "action_space" :> "sample"   :> Get '[JSON] Action
                     :<|> Capture "instance_id" InstID :> "action_space" :> "contains" :> Capture "x" Int :> Get '[JSON] Object
                     :<|> Capture "instance_id" InstID :> "observation_space"  :> Get '[JSON] Info
                     :<|> Capture "instance_id" InstID :> "monitor" :> "start" :> ReqBody '[JSON] Monitor :> Post '[HTML] ()
                     :<|> Capture "instance_id" InstID :> "monitor" :> "close" :> Post '[HTML] ()
                     :<|> Capture "instance_id" InstID :> "close"   :> Post '[HTML] ())
         :<|> "upload" :> ReqBody '[JSON] Config :> Post '[HTML] ()
         :<|> "shutdown" :> Post '[HTML] ())


-- | a proxy representing the Gym API
gymAPI :: Proxy GymAPI
gymAPI = Proxy


-- | Create an instance of the specified environment
envCreate               :: GymEnv  -> ClientM InstID
-- | List all environments running on the server
envListAll              :: ClientM Environment
-- | Reset the state of the environment and return an initial observation.
envReset                :: InstID  -> ClientM Observation
-- | Step though an environment using an action.
envStep                 :: InstID  -> Step    -> ClientM Outcome
-- | Get information (name and dimensions/bounds) of the env's `action_space`
envActionSpaceInfo      :: InstID  -> ClientM Info
-- | Get a sample from the env's action_space
envActionSpaceSample    :: InstID  -> ClientM Action
-- | Assess that value is a member of the env's action_space
envActionSpaceContains  :: InstID  -> Int     -> ClientM Object
-- | Get information (name and dimensions/bounds) of the env's `observation_space`
envObservationSpaceInfo :: InstID  -> ClientM Info
-- | Start monitoring.
envMonitorStart         :: InstID  -> Monitor -> ClientM ()
-- | Flush all monitor data to disk.
envMonitorClose         :: InstID  -> ClientM ()
-- | Manually close an environment
envClose                :: InstID  -> ClientM ()
-- | Upload the results of training (as automatically recorded by your env's monitor) to OpenAI Gym.
upload                  :: Config  -> ClientM ()
-- | Request a server shutdown
shutdownServer          :: ClientM ()


(envCreate
  :<|> envListAll
  :<|> envReset
  :<|> envStep
  :<|> envActionSpaceInfo
  :<|> envActionSpaceSample
  :<|> envActionSpaceContains
  :<|> envObservationSpaceInfo
  :<|> envMonitorStart
  :<|> envMonitorClose
  :<|> envClose)
  :<|> upload
  :<|> shutdownServer
  = client gymAPI


