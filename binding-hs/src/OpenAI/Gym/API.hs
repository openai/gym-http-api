-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.API
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveFunctor #-}
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

newtype GymClient a =
  GymClient { getGymClient :: ClientM a }
  deriving (Functor, Applicative, Monad)

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


envCreate :: EnvID -> Manager -> BaseUrl -> GymClient InstID
envCreate a b c = GymClient $ envCreate' a b c

envListAll :: Manager -> BaseUrl -> GymClient Environment
envListAll a b = GymClient $ envListAll' a b

envReset :: Text -> Manager -> BaseUrl -> GymClient Observation
envReset a b c = GymClient $ envReset' a b c

envStep :: Text -> Step -> Manager -> BaseUrl -> GymClient Outcome
envStep a b c d = GymClient $ envStep' a b c d

envActionSpaceInfo :: Text -> Manager -> BaseUrl -> GymClient Info
envActionSpaceInfo a b c = GymClient $ envActionSpaceInfo' a b c

envActionSpaceSample :: Text -> Manager -> BaseUrl -> GymClient Action
envActionSpaceSample a b c = GymClient $ envActionSpaceSample' a b c

envActionSpaceContains :: Text -> Int -> Manager -> BaseUrl -> GymClient Object
envActionSpaceContains a b c d = GymClient $ envActionSpaceContains' a b c d

envObservationSpaceInfo :: Text -> Manager -> BaseUrl -> GymClient Info
envObservationSpaceInfo a b c = GymClient $ envObservationSpaceInfo' a b c

envMonitorStart :: Text -> Monitor -> Manager -> BaseUrl -> GymClient ()
envMonitorStart a b c d = GymClient $ envMonitorStart' a b c d

envMonitorClose :: Text -> Manager -> BaseUrl -> GymClient ()
envMonitorClose a b c = GymClient $ envMonitorClose' a b c

envClose :: Text -> Manager -> BaseUrl -> GymClient ()
envClose a b c = GymClient $ envClose' a b c

upload :: Config -> Manager -> BaseUrl -> GymClient ()
upload a b c = GymClient $ upload' a b c

shutdownServer :: Manager -> BaseUrl -> GymClient ()
shutdownServer a b = GymClient $ shutdownServer' a b

