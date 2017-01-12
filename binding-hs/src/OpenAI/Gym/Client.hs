{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeOperators         #-}

module OpenAI.Gym.Client
  ( module OpenAI.Gym.Prelude
  , envCreate
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
  , upload
  , shutdownServer
  , GymEnv (..)
  , EnvID (..)
  , InstID (..)
  , Environment (..)
  , Observation (..)
  , Step (..)
  , Outcome (..)
  , Info (..)
  , Action (..)
  , Monitor (..)
  , Config (..)
  ) where

import           Data.Text          (pack)
import           OpenAI.Gym.Prelude

--------------------------------------------------------------------------------
-- DATA TYPES AND INSTANCES

data GymEnv
  -- | Classic Control Environments
  = CartPoleV0               -- ^ Balance a pole on a cart (for a short time).
  | CartPoleV1               -- ^ Balance a pole on a cart.
  | AcrobotV1                -- ^ Swing up a two-link robot.
  | MountainCarV0            -- ^ Drive up a big hill.
  | MountainCarContinuousV0  -- ^ Drive up a big hill with continuous control.
  | PendulumV0               -- ^ Swing up a pendulum.

  -- | Atari Games
  | PongRamV0                -- ^ Maximize score in the game Pong, with RAM as input
  | PongV0                   -- ^ Maximize score in the game Pong
  deriving (Eq, Enum, Ord)

instance Show GymEnv where
  show CartPoleV0              = "CartPole-v0"
  show CartPoleV1              = "CartPole-v1"
  show AcrobotV1               = "Acrobot-v1"
  show MountainCarV0           = "MountainCar-v0"
  show MountainCarContinuousV0 = "MountainCarContinuous-v0"
  show PendulumV0              = "Pendulum-v0"
  show PongRamV0               = "Pong-ram-v0"
  show PongV0                  = "Pong-v0"

instance ToJSON GymEnv where
  toJSON = String . pack . show

newtype EnvID = EnvID { env_id :: GymEnv }
  deriving Generic

instance ToJSON EnvID

newtype InstID = InstID { instance_id :: Text }
  deriving (Eq, Show, Generic)

instance ToJSON InstID
instance FromJSON InstID

newtype Environment = Environment { all_envs :: HashMap Text Text }
  deriving (Eq, Show, Generic)

instance ToJSON Environment
instance FromJSON Environment

newtype Observation = Observation { observation :: Array }
  deriving (Eq, Show, Generic)

instance ToJSON Observation
instance FromJSON Observation

data Step = Step
  { action :: !Int
  , render :: !Bool
  } deriving Generic

instance ToJSON Step

data Outcome = Outcome
  { observation :: !Array
  , reward      :: !Double
  , done        :: !Bool
  , info        :: !Object
  } deriving (Eq, Show, Generic)

instance ToJSON Outcome
instance FromJSON Outcome

newtype Info = Info { info :: Object }
  deriving (Eq, Show, Generic)

instance ToJSON Info
instance FromJSON Info

newtype Action = Action { action :: Int }
  deriving (Eq, Show, Generic)

instance ToJSON Action
instance FromJSON Action

data Monitor = Monitor
  { directory      :: !Text
  , force          :: !Bool
  , resume         :: !Bool
  , video_callable :: !Bool
  } deriving Generic

instance ToJSON Monitor

data Config = Config
  { training_dir :: !Text
  , algorithm_id :: !Text
  , api_key      :: !Text
  } deriving Generic

instance ToJSON Config

instance MimeUnrender HTML () where
    mimeUnrender _ _ = return ()

--------------------------------------------------------------------------------
-- THE API REPRESENTED AS A TYPE

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

--------------------------------------------------------------------------------
-- PROXY FOR THE API

gymAPI :: Proxy GymAPI
gymAPI = Proxy

--------------------------------------------------------------------------------
-- AUTOMATICALLY DERIVED QUERY FUNCTIONS

envCreate               :: EnvID -> Manager -> BaseUrl -> ClientM InstID
envListAll              :: Manager -> BaseUrl -> ClientM Environment
envReset                :: Text -> Manager -> BaseUrl -> ClientM Observation
envStep                 :: Text -> Step -> Manager -> BaseUrl -> ClientM Outcome
envActionSpaceInfo      :: Text -> Manager -> BaseUrl -> ClientM Info
envActionSpaceSample    :: Text -> Manager -> BaseUrl -> ClientM Action
envActionSpaceContains  :: Text -> Int -> Manager -> BaseUrl -> ClientM Object
envObservationSpaceInfo :: Text -> Manager -> BaseUrl -> ClientM Info
envMonitorStart         :: Text -> Monitor -> Manager -> BaseUrl -> ClientM ()
envMonitorClose         :: Text -> Manager -> BaseUrl -> ClientM ()
envClose                :: Text -> Manager -> BaseUrl -> ClientM ()
upload                  :: Config -> Manager -> BaseUrl -> ClientM ()
shutdownServer          :: Manager -> BaseUrl -> ClientM ()

envCreate :<|> envListAll
          :<|> envReset
          :<|> envStep
          :<|> envActionSpaceInfo
          :<|> envActionSpaceSample
          :<|> envActionSpaceContains
          :<|> envObservationSpaceInfo
          :<|> envMonitorStart
          :<|> envMonitorClose
          :<|> envClose
          :<|> upload
          :<|> shutdownServer
             = client gymAPI
