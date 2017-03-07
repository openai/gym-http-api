-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.Data
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE DeriveGeneric #-}
module OpenAI.Gym.Data
  ( GymEnv (..)
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

import OpenAI.Gym.Prelude
import qualified Data.Text  as T
import qualified Data.Aeson as A

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
  toJSON = String . T.pack . show


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


data Observation = Observation !Array
  deriving (Eq, Show, Generic)


instance ToJSON Observation where
    toJSON (Observation arr) = object ["observation" .= arr]

instance FromJSON Observation where
    parseJSON (Object v) = Observation <$> v .: "observation"
    parseJSON _          = mempty


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


data Info = Info !Object
  deriving (Eq, Show, Generic)

instance ToJSON Info where
    toJSON (Info i) = object ["info" .= i]
instance FromJSON Info where
    parseJSON (Object v) = Info <$> v .: "info"
    parseJSON _          = mempty


data Action = Action !Int
  deriving (Eq, Show, Generic)

instance ToJSON Action where
    toJSON (Action i) = object ["action" .= i]
instance FromJSON Action where
    parseJSON (Object v) = Action <$> v .: "action"
    parseJSON _          = mempty


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


