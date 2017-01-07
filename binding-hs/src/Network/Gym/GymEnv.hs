module Network.Gym.GymEnv where

import Network.Gym.Prelude
import Data.Text (pack)

type GymURL = String

defaultGymURL :: GymURL
defaultGymURL = "http://127.0.0.1:5000/v1"

class Observable res where
  observation :: res -> Vector (Vector (Vector Double))

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
  show CartPoleV0    = "CartPole-v0"
  show CartPoleV1    = "CartPole-v1"
  show AcrobotV1     = "Acrobot-v1"
  show MountainCarV0 = "MountainCar-v0"
  show MountainCarContinuousV0 = "MountainCarContinuous-v0"
  show PendulumV0    = "Pendulum-v0"
  show PongRamV0     = "Pong-ram-v0"
  show PongV0        = "Pong-v0"

instance ToJSON GymEnv where
  toJSON = String . pack . show

