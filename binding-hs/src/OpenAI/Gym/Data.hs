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

-- | a Gym environment
data GymEnv
  -- | Classic Control Environments
  = CartPoleV0               -- ^ Balance a pole on a cart (for a short time).
  | CartPoleV1               -- ^ Balance a pole on a cart.
  | AcrobotV1                -- ^ Swing up a two-link robot.
  | MountainCarV0            -- ^ Drive up a big hill.
  | MountainCarContinuousV0  -- ^ Drive up a big hill with continuous control.
  | PendulumV0               -- ^ Swing up a pendulum.

  -- | Toy text games
  | FrozenLakeV0             -- ^ Swing up a pendulum.

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
  show FrozenLakeV0            = "FrozenLake-v0"
  show PongRamV0               = "Pong-ram-v0"
  show PongV0                  = "Pong-v0"

instance ToJSON GymEnv where
  toJSON env = object [ "env_id" .= show env ]


-- | instance identifyier for the environment
data InstID = InstID !Text
  deriving (Eq, Show, Generic)

instance ToHttpApiData InstID where
  toUrlPiece (InstID i) = i

instance ToJSON InstID where
  toJSON (InstID i) = toSingleton "instance_id" i

instance FromJSON InstID where
  parseJSON = parseSingleton InstID "instance_id"


-- | dict mapping `instance_id` to `env_id` (e.g. `{'3c657dbc': 'CartPole-v0'}`)
-- for every env on the server
newtype Environment = Environment { all_envs :: HashMap Text Text }
  deriving (Eq, Show, Generic)

instance ToJSON Environment
instance FromJSON Environment


-- | an observation of the environment, conforming to the environment's
-- observation space
data Observation = Observation !Value
  deriving (Eq, Show, Generic)

instance ToJSON Observation where
  toJSON (Observation v) = toSingleton "observation" v

instance FromJSON Observation where
  parseJSON = parseSingleton Observation "observation"


-- | settings for the environment's `step`
data Step = Step
  { action :: !Value
  , render :: !Bool
  } deriving (Eq, Generic, Show)

instance ToJSON Step


-- | the outcome of an action in the environment
data Outcome = Outcome
  { observation :: !Value
  , reward      :: !Double
  , done        :: !Bool
  , info        :: !Object
  } deriving (Eq, Show, Generic)

instance ToJSON Outcome
instance FromJSON Outcome


-- | additional info provided by the environment for debugging purposes
data Info = Info !Object
  deriving (Eq, Show, Generic)

instance ToJSON Info where
  toJSON (Info v) = toSingleton "info" v

instance FromJSON Info where
  parseJSON = parseSingleton Info "info"


-- | an action to take in the environment, constrained by the environment's
-- action space
data Action = Action !Value
  deriving (Eq, Show, Generic)

instance ToJSON Action where
  toJSON (Action v) = toSingleton "action" v

instance FromJSON Action where
  parseJSON = parseSingleton Action "action"


-- | configurations for monitoring the environment
data Monitor = Monitor
  { directory      :: !Text -- ^ Directory to dump the monitoring files.
  , force          :: !Bool -- ^ Clear out existing training data from this
                            -- directory (by deleting every file prefixed
                            -- with "openaigym.")
  , resume         :: !Bool -- ^ Retain the training data already in this
                            -- directory, which will be merged with our new data
  , video_callable :: !Bool -- ^ Whether to record a video on this episode.
                            -- Not yet implemented in the HTTP API, so fixed to
                            -- false to disable video recording. Otherwise for
                            -- null takes perfect cubes, capped at 1000.
  } deriving (Generic, Eq, Show)

instance ToJSON Monitor


-- | settings for uploading run data to OpenAI Gym
data Config = Config
  { training_dir :: !Text -- ^ training_dir: A directory containing the results
                          -- of a training run.
  , algorithm_id :: !Text -- ^ algorithm_id (default=None): An arbitrary string
                          -- indicating the paricular version of the algorithm
                          -- (including choices of parameters) you are running.
  , api_key      :: !Text -- ^ api_key: Your OpenAI API key
  } deriving (Generic, Eq, Show)

instance ToJSON Config


