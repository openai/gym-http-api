-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.Data
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
--
-- Aeson-based data types to be returned by "OpenAI.Gym.API"
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

-- | Game Environments
data GymEnv
  -- Classic Control Environments
  = CartPoleV0               -- ^ Balance a pole on a cart (for a short time).
  | CartPoleV1               -- ^ Balance a pole on a cart.
  | AcrobotV1                -- ^ Swing up a two-link robot.
  | MountainCarV0            -- ^ Drive up a big hill.
  | MountainCarContinuousV0  -- ^ Drive up a big hill with continuous control.
  | PendulumV0               -- ^ Swing up a pendulum.

  -- Toy text games
  | BlackjackV0     -- ^ Play Blackjack against a computer dealer
  | FrozenLakeV0    -- ^ Find a safe path across a grid of ice and water tiles.
  | FrozenLake8x8V0 -- ^ Find a safe path across a grid of ice and water tiles.
  | GuessingGameV0  -- ^ Guess close to randomly selected number
  | HotterColderV0  -- ^ Guess close to a random selected number using hints
  | NChainV0        -- ^ Traverse a linear chain of states
  | RouletteV0      -- ^ Learn a winning strategy for playing roulette.
  | TaxiV2          -- ^ As a taxi driver, you need to pick up and drop off passengers as fast as possible.

  -- Box2D
  | BipedalWalkerV2         -- ^ Train a bipedal robot to walk.
  | BipedalWalkerHardcoreV2 -- ^ Train a bipedal robot to walk over rough terrain.
  | CarRacingV0             -- ^ Race a car around a track.
  | LunarLanderV2           -- ^ Navigate a lander to its landing pad.
  | LunarLanderContinuousV2 -- ^ Navigate a lander to its landing pad.

  -- Algorithms
  | CopyV0              -- ^ Copy symbols from the input tape.
  | DuplicatedInputV0   -- ^  Copy and deduplicate data from the input tape.
  | RepeatCopyV0        -- ^  Copy symbols from the input tape multiple times.
  | ReverseV0           -- ^  Reverse the symbols on the input tape.
  | ReversedAdditionV0  -- ^  Learn to add multi-digit numbers.
  | ReversedAddition3V0 -- ^ Learn to add three multi-digit numbers.

  -- Atari Games
  | AirRaidRamV0          -- ^ Maximize score in the game AirRaid, with RAM as input
  | AirRaidV0             -- ^ Maximize score in the game AirRaid, with screen images as input
  | AlienRamV0            -- ^ Maximize score in the game Alien, with RAM as input
  | AlienV0               -- ^ Maximize score in the game Alien, with screen images as input
  | AmidarRamV0           -- ^ Maximize score in the game Amidar, with RAM as input
  | AmidarV0              -- ^ Maximize score in the game Amidar, with screen images as input
  | AssaultRamV0          -- ^ Maximize score in the game Assault, with RAM as input
  | AssaultV0             -- ^ Maximize score in the game Assault, with screen images as input
  | AsterixRamV0          -- ^ Maximize score in the game Asterix, with RAM as input
  | AsterixV0             -- ^ Maximize score in the game Asterix, with screen images as input
  | AsteroidsRamV0        -- ^ Maximize score in the game Asteroids, with RAM as input
  | AsteroidsV0           -- ^ Maximize score in the game Asteroids, with screen images as input
  | AtlantisRamV0         -- ^ Maximize score in the game Atlantis, with RAM as input
  | AtlantisV0            -- ^ Maximize score in the game Atlantis, with screen images as input
  | BankHeistRamV0        -- ^ Maximize score in the game BankHeist, with RAM as input
  | BankHeistV0           -- ^ Maximize score in the game BankHeist, with screen images as input
  | BattleZoneRamV0       -- ^ Maximize score in the game BattleZone, with RAM as input
  | BattleZoneV0          -- ^ Maximize score in the game BattleZone, with screen images as input
  | BeamRiderRamV0        -- ^ Maximize score in the game BeamRider, with RAM as input
  | BeamRiderV0           -- ^ Maximize score in the game BeamRider, with screen images as input
  | BerzerkRamV0          -- ^ Maximize score in the game Berzerk, with RAM as input
  | BerzerkV0             -- ^ Maximize score in the game Berzerk, with screen images as input
  | BowlingRamV0          -- ^ Maximize score in the game Bowling, with RAM as input
  | BowlingV0             -- ^ Maximize score in the game Bowling, with screen images as input
  | BoxingRamV0           -- ^ Maximize score in the game Boxing, with RAM as input
  | BoxingV0              -- ^ Maximize score in the game Boxing, with screen images as input
  | BreakoutRamV0         -- ^ Maximize score in the game Breakout, with RAM as input
  | BreakoutV0            -- ^ Maximize score in the game Breakout, with screen images as input
  | CarnivalRamV0         -- ^ Maximize score in the game Carnival, with RAM as input
  | CarnivalV0            -- ^ Maximize score in the game Carnival, with screen images as input
  | CentipedeRamV0        -- ^ Maximize score in the game Centipede, with RAM as input
  | CentipedeV0           -- ^ Maximize score in the game Centipede, with screen images as input
  | ChopperCommandRamV0   -- ^ Maximize score in the game ChopperCommand, with RAM as input
  | ChopperCommandV0      -- ^ Maximize score in the game ChopperCommand, with screen images as input
  | CrazyClimberRamV0     -- ^ Maximize score in the game CrazyClimber, with RAM as input
  | CrazyClimberV0        -- ^ Maximize score in the game CrazyClimber, with screen images as input
  | DemonAttackRamV0      -- ^ Maximize score in the game DemonAttack, with RAM as input
  | DemonAttackV0         -- ^ Maximize score in the game DemonAttack, with screen images as input
  | DoubleDunkRamV0       -- ^ Maximize score in the game DoubleDunk, with RAM as input
  | DoubleDunkV0          -- ^ Maximize score in the game DoubleDunk, with screen images as input
  | ElevatorActionRamV0   -- ^ Maximize score in the game ElevatorAction, with RAM as input
  | ElevatorActionV0      -- ^ Maximize score in the game ElevatorAction, with screen images as input
  | EnduroRamV0           -- ^ Maximize score in the game Enduro, with RAM as input
  | EnduroV0              -- ^ Maximize score in the game Enduro, with screen images as input
  | FishingDerbyRamV0     -- ^ Maximize score in the game FishingDerby, with RAM as input
  | FishingDerbyV0        -- ^ Maximize score in the game FishingDerby, with screen images as input
  | FreewayRamV0          -- ^ Maximize score in the game Freeway, with RAM as input
  | FreewayV0             -- ^ Maximize score in the game Freeway, with screen images as input
  | FrostbiteRamV0        -- ^ Maximize score in the game Frostbite, with RAM as input
  | FrostbiteV0           -- ^ Maximize score in the game Frostbite, with screen images as input
  | GopherRamV0           -- ^ Maximize score in the game Gopher, with RAM as input
  | GopherV0              -- ^ Maximize score in the game Gopher, with screen images as input
  | GravitarRamV0         -- ^ Maximize score in the game Gravitar, with RAM as input
  | GravitarV0            -- ^ Maximize score in the game Gravitar, with screen images as input
  | IceHockeyRamV0        -- ^ Maximize score in the game IceHockey, with RAM as input
  | IceHockeyV0           -- ^ Maximize score in the game IceHockey, with screen images as input
  | JamesbondRamV0        -- ^ Maximize score in the game Jamesbond, with RAM as input
  | JamesbondV0           -- ^ Maximize score in the game Jamesbond, with screen images as input
  | JourneyEscapeRamV0    -- ^ Maximize score in the game JourneyEscape, with RAM as input
  | JourneyEscapeV0       -- ^ Maximize score in the game JourneyEscape, with screen images as input
  | KangarooRamV0         -- ^ Maximize score in the game Kangaroo, with RAM as input
  | KangarooV0            -- ^ Maximize score in the game Kangaroo, with screen images as input
  | KrullRamV0            -- ^ Maximize score in the game Krull, with RAM as input
  | KrullV0               -- ^ Maximize score in the game Krull, with screen images as input
  | KungFuMasterRamV0     -- ^ Maximize score in the game KungFuMaster, with RAM as input
  | KungFuMasterV0        -- ^ Maximize score in the game KungFuMaster, with screen images as input
  | MontezumaRevengeRamV0 -- ^ Maximize score in the game MontezumaRevenge, with RAM as input
  | MontezumaRevengeV0    -- ^ Maximize score in the game MontezumaRevenge, with screen images as input
  | MsPacmanRamV0         -- ^ Maximize score in the game MsPacman, with RAM as input
  | MsPacmanV0            -- ^ Maximize score in the game MsPacman, with screen images as input
  | NameThisGameRamV0     -- ^ Maximize score in the game NameThisGame, with RAM as input
  | NameThisGameV0        -- ^ Maximize score in the game NameThisGame, with screen images as input
  | PhoenixRamV0          -- ^ Maximize score in the game Phoenix, with RAM as input
  | PhoenixV0             -- ^ Maximize score in the game Phoenix, with screen images as input
  | PitfallRamV0          -- ^ Maximize score in the game Pitfall, with RAM as input
  | PitfallV0             -- ^ Maximize score in the game Pitfall, with screen images as input
  | PongRamV0             -- ^ Maximize score in the game Pong, with RAM as input
  | PongV0                -- ^ Maximize score in the game Pong, with screen images as input
  | PooyanRamV0           -- ^ Maximize score in the game Pooyan, with RAM as input
  | PooyanV0              -- ^ Maximize score in the game Pooyan, with screen images as input
  | PrivateEyeRamV0       -- ^ Maximize score in the game PrivateEye, with RAM as input
  | PrivateEyeV0          -- ^ Maximize score in the game PrivateEye, with screen images as input
  | QbertRamV0            -- ^ Maximize score in the game Qbert, with RAM as input
  | QbertV0               -- ^ Maximize score in the game Qbert, with screen images as input
  | RiverraidRamV0        -- ^ Maximize score in the game Riverraid, with RAM as input
  | RiverraidV0           -- ^ Maximize score in the game Riverraid, with screen images as input
  | RoadRunnerRamV0       -- ^ Maximize score in the game RoadRunner, with RAM as input
  | RoadRunnerV0          -- ^ Maximize score in the game RoadRunner, with screen images as input
  | RobotankRamV0         -- ^ Maximize score in the game Robotank, with RAM as input
  | RobotankV0            -- ^ Maximize score in the game Robotank, with screen images as input
  | SeaquestRamV0         -- ^ Maximize score in the game Seaquest, with RAM as input
  | SeaquestV0            -- ^ Maximize score in the game Seaquest, with screen images as input
  | SkiingRamV0           -- ^ Maximize score in the game Skiing, with RAM as input
  | SkiingV0              -- ^ Maximize score in the game Skiing, with screen images as input
  | SolarisRamV0          -- ^ Maximize score in the game Solaris, with RAM as input
  | SolarisV0             -- ^ Maximize score in the game Solaris, with screen images as input
  | SpaceInvadersRamV0    -- ^ Maximize score in the game SpaceInvaders, with RAM as input
  | SpaceInvadersV0       -- ^ Maximize score in the game SpaceInvaders, with screen images as input
  | StarGunnerRamV0       -- ^ Maximize score in the game StarGunner, with RAM as input
  | StarGunnerV0          -- ^ Maximize score in the game StarGunner, with screen images as input
  | TennisRamV0           -- ^ Maximize score in the game Tennis, with RAM as input
  | TennisV0              -- ^ Maximize score in the game Tennis, with screen images as input
  | TimePilotRamV0        -- ^ Maximize score in the game TimePilot, with RAM as input
  | TimePilotV0           -- ^ Maximize score in the game TimePilot, with screen images as input
  | TutankhamRamV0        -- ^ Maximize score in the game Tutankham, with RAM as input
  | TutankhamV0           -- ^ Maximize score in the game Tutankham, with screen images as input
  | UpNDownRamV0          -- ^ Maximize score in the game UpNDown, with RAM as input
  | UpNDownV0             -- ^ Maximize score in the game UpNDown, with screen images as input
  | VentureRamV0          -- ^ Maximize score in the game Venture, with RAM as input
  | VentureV0             -- ^ Maximize score in the game Venture, with screen images as input
  | VideoPinballRamV0     -- ^ Maximize score in the game VideoPinball, with RAM as input
  | VideoPinballV0        -- ^ Maximize score in the game VideoPinball, with screen images as input
  | WizardOfWorRamV0      -- ^ Maximize score in the game WizardOfWor, with RAM as input
  | WizardOfWorV0         -- ^ Maximize score in the game WizardOfWor, with screen images as input
  | YarsRevengeRamV0      -- ^ Maximize score in the game YarsRevenge, with RAM as input
  | YarsRevengeV0         -- ^ Maximize score in the game YarsRevenge, with screen images as input
  | ZaxxonRamV0           -- ^ Maximize score in the game Zaxxon, with RAM as input
  | ZaxxonV0              -- ^ Maximize score in the game Zaxxon, with screen images as input
  deriving (Eq, Enum, Ord)

instance Show GymEnv where
  show = \case
    CartPoleV0              -> "CartPole-v0"
    CartPoleV1              -> "CartPole-v1"
    AcrobotV1               -> "Acrobot-v1"
    MountainCarV0           -> "MountainCar-v0"
    MountainCarContinuousV0 -> "MountainCarContinuous-v0"
    PendulumV0              -> "Pendulum-v0"

    BlackjackV0     -> "Blackjack-v0"
    FrozenLakeV0    -> "FrozenLake-v0"
    FrozenLake8x8V0 -> "FrozenLake8x8-v0"
    GuessingGameV0  -> "GuessingGame-v0"
    HotterColderV0  -> "HotterColder-v0"
    NChainV0        -> "NChain-v0"
    RouletteV0      -> "Roulette-v0"
    TaxiV2          -> "Taxi-v2"

    BipedalWalkerV2         -> "BipedalWalker-v2"
    BipedalWalkerHardcoreV2 -> "BipedalWalkerHardcore-v2"
    CarRacingV0             -> "CarRacing-v0"
    LunarLanderV2           -> "LunarLander-v2"
    LunarLanderContinuousV2 -> "LunarLanderContinuous-v2"

    CopyV0              -> "Copy-v0"
    DuplicatedInputV0   -> "DuplicatedInput-v0"
    RepeatCopyV0        -> "RepeatCopy-v0"
    ReverseV0           -> "Reverse-v0"
    ReversedAdditionV0  -> "ReversedAddition-v0"
    ReversedAddition3V0 -> "ReversedAddition3-v0"

    AirRaidRamV0          -> "AirRaid-ram-v0"
    AirRaidV0             -> "AirRaid-v0"
    AlienRamV0            -> "Alien-ram-v0"
    AlienV0               -> "Alien-v0"
    AmidarRamV0           -> "Amidar-ram-v0"
    AmidarV0              -> "Amidar-v0"
    AssaultRamV0          -> "Assault-ram-v0"
    AssaultV0             -> "Assault-v0"
    AsterixRamV0          -> "Asterix-ram-v0"
    AsterixV0             -> "Asterix-v0"
    AsteroidsRamV0        -> "Asteroids-ram-v0"
    AsteroidsV0           -> "Asteroids-v0"
    AtlantisRamV0         -> "Atlantis-ram-v0"
    AtlantisV0            -> "Atlantis-v0"
    BankHeistRamV0        -> "BankHeist-ram-v0"
    BankHeistV0           -> "BankHeist-v0"
    BattleZoneRamV0       -> "BattleZone-ram-v0"
    BattleZoneV0          -> "BattleZone-v0"
    BeamRiderRamV0        -> "BeamRider-ram-v0"
    BeamRiderV0           -> "BeamRider-v0"
    BerzerkRamV0          -> "Berzerk-ram-v0"
    BerzerkV0             -> "Berzerk-v0"
    BowlingRamV0          -> "Bowling-ram-v0"
    BowlingV0             -> "Bowling-v0"
    BoxingRamV0           -> "Boxing-ram-v0"
    BoxingV0              -> "Boxing-v0"
    BreakoutRamV0         -> "Breakout-ram-v0"
    BreakoutV0            -> "Breakout-v0"
    CarnivalRamV0         -> "Carnival-ram-v0"
    CarnivalV0            -> "Carnival-v0"
    CentipedeRamV0        -> "Centipede-ram-v0"
    CentipedeV0           -> "Centipede-v0"
    ChopperCommandRamV0   -> "ChopperCommand-ram-v0"
    ChopperCommandV0      -> "ChopperCommand-v0"
    CrazyClimberRamV0     -> "CrazyClimber-ram-v0"
    CrazyClimberV0        -> "CrazyClimber-v0"
    DemonAttackRamV0      -> "DemonAttack-ram-v0"
    DemonAttackV0         -> "DemonAttack-v0"
    DoubleDunkRamV0       -> "DoubleDunk-ram-v0"
    DoubleDunkV0          -> "DoubleDunk-v0"
    ElevatorActionRamV0   -> "ElevatorAction-ram-v0"
    ElevatorActionV0      -> "ElevatorAction-v0"
    EnduroRamV0           -> "Enduro-ram-v0"
    EnduroV0              -> "Enduro-v0"
    FishingDerbyRamV0     -> "FishingDerby-ram-v0"
    FishingDerbyV0        -> "FishingDerby-v0"
    FreewayRamV0          -> "Freeway-ram-v0"
    FreewayV0             -> "Freeway-v0"
    FrostbiteRamV0        -> "Frostbite-ram-v0"
    FrostbiteV0           -> "Frostbite-v0"
    GopherRamV0           -> "Gopher-ram-v0"
    GopherV0              -> "Gopher-v0"
    GravitarRamV0         -> "Gravitar-ram-v0"
    GravitarV0            -> "Gravitar-v0"
    IceHockeyRamV0        -> "IceHockey-ram-v0"
    IceHockeyV0           -> "IceHockey-v0"
    JamesbondRamV0        -> "Jamesbond-ram-v0"
    JamesbondV0           -> "Jamesbond-v0"
    JourneyEscapeRamV0    -> "JourneyEscape-ram-v0"
    JourneyEscapeV0       -> "JourneyEscape-v0"
    KangarooRamV0         -> "Kangaroo-ram-v0"
    KangarooV0            -> "Kangaroo-v0"
    KrullRamV0            -> "Krull-ram-v0"
    KrullV0               -> "Krull-v0"
    KungFuMasterRamV0     -> "KungFuMaster-ram-v0"
    KungFuMasterV0        -> "KungFuMaster-v0"
    MontezumaRevengeRamV0 -> "MontezumaRevenge-ram-v0"
    MontezumaRevengeV0    -> "MontezumaRevenge-v0"
    MsPacmanRamV0         -> "MsPacman-ram-v0"
    MsPacmanV0            -> "MsPacman-v0"
    NameThisGameRamV0     -> "NameThisGame-ram-v0"
    NameThisGameV0        -> "NameThisGame-v0"
    PhoenixRamV0          -> "Phoenix-ram-v0"
    PhoenixV0             -> "Phoenix-v0"
    PitfallRamV0          -> "Pitfall-ram-v0"
    PitfallV0             -> "Pitfall-v0"
    PongRamV0             -> "Pong-ram-v0"
    PongV0                -> "Pong-v0"
    PooyanRamV0           -> "Pooyan-ram-v0"
    PooyanV0              -> "Pooyan-v0"
    PrivateEyeRamV0       -> "PrivateEye-ram-v0"
    PrivateEyeV0          -> "PrivateEye-v0"
    QbertRamV0            -> "Qbert-ram-v0"
    QbertV0               -> "Qbert-v0"
    RiverraidRamV0        -> "Riverraid-ram-v0"
    RiverraidV0           -> "Riverraid-v0"
    RoadRunnerRamV0       -> "RoadRunner-ram-v0"
    RoadRunnerV0          -> "RoadRunner-v0"
    RobotankRamV0         -> "Robotank-ram-v0"
    RobotankV0            -> "Robotank-v0"
    SeaquestRamV0         -> "Seaquest-ram-v0"
    SeaquestV0            -> "Seaquest-v0"
    SkiingRamV0           -> "Skiing-ram-v0"
    SkiingV0              -> "Skiing-v0"
    SolarisRamV0          -> "Solaris-ram-v0"
    SolarisV0             -> "Solaris-v0"
    SpaceInvadersRamV0    -> "SpaceInvaders-ram-v0"
    SpaceInvadersV0       -> "SpaceInvaders-v0"
    StarGunnerRamV0       -> "StarGunner-ram-v0"
    StarGunnerV0          -> "StarGunner-v0"
    TennisRamV0           -> "Tennis-ram-v0"
    TennisV0              -> "Tennis-v0"
    TimePilotRamV0        -> "TimePilot-ram-v0"
    TimePilotV0           -> "TimePilot-v0"
    TutankhamRamV0        -> "Tutankham-ram-v0"
    TutankhamV0           -> "Tutankham-v0"
    UpNDownRamV0          -> "UpNDown-ram-v0"
    UpNDownV0             -> "UpNDown-v0"
    VentureRamV0          -> "Venture-ram-v0"
    VentureV0             -> "Venture-v0"
    VideoPinballRamV0     -> "VideoPinball-ram-v0"
    VideoPinballV0        -> "VideoPinball-v0"
    WizardOfWorRamV0      -> "WizardOfWor-ram-v0"
    WizardOfWorV0         -> "WizardOfWor-v0"
    YarsRevengeRamV0      -> "YarsRevenge-ram-v0"
    YarsRevengeV0         -> "YarsRevenge-v0"
    ZaxxonRamV0           -> "Zaxxon-ram-v0"
    ZaxxonV0              -> "Zaxxon-v0"

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

