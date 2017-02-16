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

type GymAPI = "v1" :> "envs" :> ReqBody '[JSON] EnvID :> Post '[JSON] InstID
         :<|> "v1" :> "envs" :> Get '[JSON] Environment
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "reset" :> Post '[JSON] Observation
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> ReqBody '[JSON] Step :> "step" :> Post '[JSON] Outcome
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "action_space" :> Get '[JSON] Info
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "action_space" :> "sample" :> Get '[JSON] Action
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "action_space" :> "contains" :> Capture "x" Int
              :> Get '[JSON] Object
         :<|> "v1" :> "envs"
              :> Capture "instance_od" Text :> "observation_space"
              :> Get '[JSON] Info
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text :> ReqBody '[JSON] Monitor
              :> "monitor" :> "start" :> Post '[HTML] ()
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "monitor" :> "close" :> Post '[HTML] ()
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text :> "close"
              :> Post '[HTML] ()
         :<|> "v1" :> "upload" :> ReqBody '[JSON] Config :> Post '[HTML] ()
         :<|> "v1" :> "shutdown" :> Post '[HTML] ()


gymAPI :: Proxy GymAPI
gymAPI = Proxy


envCreate'               :: EnvID   -> Manager -> BaseUrl -> ClientM InstID
envListAll'              :: Manager -> BaseUrl -> ClientM Environment
envReset'                :: Text    -> Manager -> BaseUrl -> ClientM Observation
envStep'                 :: Text    -> Step    -> Manager -> BaseUrl -> ClientM Outcome
envActionSpaceInfo'      :: Text    -> Manager -> BaseUrl -> ClientM Info
envActionSpaceSample'    :: Text    -> Manager -> BaseUrl -> ClientM Action
envActionSpaceContains'  :: Text    -> Int     -> Manager -> BaseUrl -> ClientM Object
envObservationSpaceInfo' :: Text    -> Manager -> BaseUrl -> ClientM Info
envMonitorStart'         :: Text    -> Monitor -> Manager -> BaseUrl -> ClientM ()
envMonitorClose'         :: Text    -> Manager -> BaseUrl -> ClientM ()
envClose'                :: Text    -> Manager -> BaseUrl -> ClientM ()
upload'                  :: Config  -> Manager -> BaseUrl -> ClientM ()
shutdownServer'          :: Manager -> BaseUrl -> ClientM ()


envCreate'
  :<|> envListAll'
  :<|> envReset'
  :<|> envStep'
  :<|> envActionSpaceInfo'
  :<|> envActionSpaceSample'
  :<|> envActionSpaceContains'
  :<|> envObservationSpaceInfo'
  :<|> envMonitorStart'
  :<|> envMonitorClose'
  :<|> envClose'
  :<|> upload'
  :<|> shutdownServer'
  = client gymAPI


