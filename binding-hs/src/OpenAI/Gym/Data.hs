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

import Data.Aeson (ToJSON(..), FromJSON(..), Value(..), Object, (.=), (.:), object)
import Data.Aeson.Types (Parser)
import Data.HashMap.Strict (HashMap)
import Data.Text (Text)
import GHC.Generics (Generic)
import Servant.API (ToHttpApiData(..))
import qualified Data.Text  as T
import qualified Data.Aeson as A

-- | Classic Control Environments
data GymEnv
  = CartPoleV0               -- ^ Balance a pole on a cart (for a short time).
  | CartPoleV1               -- ^ Balance a pole on a cart.
  | AcrobotV1                -- ^ Swing up a two-link robot.
  | MountainCarV0            -- ^ Drive up a big hill.
  | MountainCarContinuousV0  -- ^ Drive up a big hill with continuous control.
  | PendulumV0               -- ^ Swing up a pendulum.

  -- Toy text games
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

-- | a short identifier (such as '3c657dbc') for the created environment instance.
-- The instance_id is used in future API calls to identify the environment to be manipulated.
newtype InstID = InstID { getInstID :: Text }
  deriving (Eq, Show, Generic)

instance ToHttpApiData InstID where
  toUrlPiece (InstID i) = i

instance ToJSON InstID where
  toJSON (InstID i) = toSingleton "instance_id" i

instance FromJSON InstID where
  parseJSON = parseSingleton InstID "instance_id"

-- | a mapping of instance_id to env_id (e.g. {'3c657dbc': 'CartPole-v0'}) for every env on the server
newtype Environment = Environment { all_envs :: HashMap Text Text }
  deriving (Eq, Show, Generic)

instance ToJSON Environment
instance FromJSON Environment

-- | The agent's observation of the current environment
newtype Observation = Observation { getObservation :: Value }
  deriving (Eq, Show, Generic)

instance ToJSON Observation where
  toJSON (Observation v) = toSingleton "observation" v

instance FromJSON Observation where
  parseJSON = parseSingleton Observation "observation"

-- | An action to take in the environment and whether or not to render that change
data Step = Step
  { action :: !Value
  , render :: !Bool
  } deriving (Eq, Generic, Show)

instance ToJSON Step

-- | The result of taking a step in an environment
data Outcome = Outcome
  { observation :: !Value  -- ^ agent's observation of the current environment
  , reward      :: !Double -- ^ amount of reward returned after previous action
  , done        :: !Bool   -- ^ whether the episode has ended
  , info        :: !Object -- ^ a dict containing auxiliary diagnostic information
  } deriving (Eq, Show, Generic)

instance ToJSON Outcome
instance FromJSON Outcome

-- | A dict containing auxiliary diagnostic information
newtype Info = Info { getInfo :: Object }
  deriving (Eq, Show, Generic)

instance ToJSON Info where
  toJSON (Info v) = toSingleton "info" v

instance FromJSON Info where
  parseJSON = parseSingleton Info "info"

-- | An action to take in the environment
newtype Action = Action { getAction :: Value }
  deriving (Eq, Show, Generic)

instance ToJSON Action where
  toJSON (Action v) = toSingleton "action" v

instance FromJSON Action where
  parseJSON = parseSingleton Action "action"

-- | Parameters used to start a monitoring session.
data Monitor = Monitor
  { directory      :: !Text -- ^ directory to use for monitoring
  , force          :: !Bool -- ^ Clear out existing training data from this directory (by deleting
                            --   every file prefixed with "openaigym.") (default=False)
  , resume         :: !Bool -- ^ Retain the training data already in this directory, which will be
                            --   merged with our new data. (default=False)
  , video_callable :: !Bool -- ^ video_callable parameter from the native env.monitor.start function
  } deriving (Generic, Eq, Show)

instance ToJSON Monitor

-- | Parameters used to upload a monitored session to OpenAI's servers
data Config = Config
  { training_dir :: !Text -- ^ A directory containing the results of a training run.
  , algorithm_id :: !Text -- ^ An arbitrary string indicating the paricular version of the
                          --   algorithm (including choices of parameters) you are running.
                          --   (default=None)
  , api_key      :: !Text -- ^ Your OpenAI API key
  } deriving (Generic, Eq, Show)

instance ToJSON Config


-- | helper to parse a singleton object from aeson
parseSingleton :: FromJSON a => (a -> b) -> Text -> Value -> Parser b
parseSingleton fn f (Object v) = fn <$> v .: f
parseSingleton fn f _          = mempty

-- | convert a value into a singleton object
toSingleton :: ToJSON a => Text -> a -> Value
toSingleton f a = object [ f .= toJSON a ]

