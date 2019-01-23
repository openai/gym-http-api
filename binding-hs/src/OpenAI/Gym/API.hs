-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.API
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
--
-- Servant-client functions to interact with the flask server from
-- <https://github.com/openai/gym-http-api/ openai/gym-http-api>.
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wno-orphans #-} --for MimeUnrender HTML ()
module OpenAI.Gym.API (
  -- * Environment functions
    envCreate
  , envListAll
  , envReset
  , envStep
  , envActionSpaceInfo
  , envActionSpaceSample
  , envActionSpaceContains
  , envObservationSpaceInfo
  , envMonitorStart
  , envMonitorClose
  , envClose
  -- * Http-server commands
  , upload
  , shutdownServer
  -- * Servant code
  , gymAPI
  ) where

import Data.Aeson (Object)
import Data.Proxy (Proxy(..))
import Servant.API
import Servant.HTML.Lucid (HTML)
import Servant.Client (ClientM, client)

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


-- | Proxy type for the full servant-client representation of the gym http api.
gymAPI :: Proxy GymAPI
gymAPI = Proxy


-- | Create an instance of the specified environment (@POST \/v1\/envs\/@)
envCreate :: GymEnv -> ClientM InstID

-- | List all environments running on the server (@GET \/v1\/envs\/@)
envListAll :: ClientM Environment

-- | Reset the state of the environment and return an initial observation. (@POST \/v1\/envs\/<instance_id>\/reset\/@)
envReset :: InstID -> ClientM Observation

-- | Step though an environment using an action. (@POST \/v1\/envs\/<instance_id>\/step\/@)
envStep :: InstID -> Step -> ClientM Outcome

-- | Get information (name and dimensions\/bounds) of the env's action_space (@GET \/v1\/envs\/<instance_id>\/action_space\/@)
envActionSpaceInfo :: InstID -> ClientM Info

-- | Sample randomly from the env's action_space (@GET \/v1\/envs\/<instance_id>\/action_space\/sample@)
envActionSpaceSample :: InstID -> ClientM Action

-- | Check to see if a value is valid in the env's action_space (@GET \/v1\/envs\/<instance_id>\/action_space\/contains\/<x>@)
envActionSpaceContains :: InstID -> Int -> ClientM Object

-- | Get information (name and dimensions\/bounds) of the env's observation_space (@GET \/v1\/envs\/<instance_id>\/observation_space\/@)
envObservationSpaceInfo :: InstID -> ClientM Info

-- | Start monitoring (@POST \/v1\/envs\/<instance_id>\/monitor\/start\/@)
envMonitorStart :: InstID -> Monitor -> ClientM ()

-- | Flush all monitor data to disk (@POST \/v1\/envs\/<instance_id>\/monitor\/close\/@)
envMonitorClose :: InstID -> ClientM ()

-- | Stop the environment (@POST \/v1\/envs\/<instance_id>\/close\/@)
envClose :: InstID -> ClientM ()

-- | Upload results to OpenAI's servers (@POST \/v1\/upload\/@)
upload :: Config -> ClientM ()

-- | Request a server shutdown (@POST \/v1\/shutdown\/@)
shutdownServer :: ClientM ()


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

instance MimeUnrender HTML () where
    mimeUnrender _ _ = return ()
