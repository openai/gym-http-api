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


gymAPI :: Proxy GymAPI
gymAPI = Proxy


envCreate'               :: GymEnv  -> Manager -> BaseUrl -> ClientM InstID
envListAll'              :: Manager -> BaseUrl -> ClientM Environment
envReset'                :: InstID  -> Manager -> BaseUrl -> ClientM Observation
envStep'                 :: InstID  -> Step    -> Manager -> BaseUrl -> ClientM Outcome
envActionSpaceInfo'      :: InstID  -> Manager -> BaseUrl -> ClientM Info
envActionSpaceSample'    :: InstID  -> Manager -> BaseUrl -> ClientM Action
envActionSpaceContains'  :: InstID  -> Int     -> Manager -> BaseUrl -> ClientM Object
envObservationSpaceInfo' :: InstID  -> Manager -> BaseUrl -> ClientM Info
envMonitorStart'         :: InstID  -> Monitor -> Manager -> BaseUrl -> ClientM ()
envMonitorClose'         :: InstID  -> Manager -> BaseUrl -> ClientM ()
envClose'                :: InstID  -> Manager -> BaseUrl -> ClientM ()
upload'                  :: Config  -> Manager -> BaseUrl -> ClientM ()
shutdownServer'          :: Manager -> BaseUrl -> ClientM ()


(envCreate'
  :<|> envListAll'
  :<|> envReset'
  :<|> envStep'
  :<|> envActionSpaceInfo'
  :<|> envActionSpaceSample'
  :<|> envActionSpaceContains'
  :<|> envObservationSpaceInfo'
  :<|> envMonitorStart'
  :<|> envMonitorClose'
  :<|> envClose')
  :<|> upload'
  :<|> shutdownServer'
  = client gymAPI


