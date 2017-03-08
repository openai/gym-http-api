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


