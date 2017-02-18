-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.Client
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveFunctor #-}
module OpenAI.Gym.Client
  ( module X -- TODO: this module is a little messy. This should just be reexports to X
  , GymClient(..)
  , runGymClient
  , getConnection
  , withConnection
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
  ) where

import OpenAI.Gym.Data as X

-- minimal re-exports for any client dependencies
import Data.Aeson as X
import Control.Monad.Trans.Except as X (runExceptT)
import Network.HTTP.Client as X
  ( Manager(..)
  , newManager
  , defaultManagerSettings
  )
import Servant.Client as X
  ( BaseUrl(..)
  , Scheme(..)
  )

-- ========================================================================= --

import OpenAI.Gym.API
import OpenAI.Gym.Prelude


-- | GymClient is our primary computational context
newtype GymClient a =
  GymClient { getGymClient :: ReaderT (Manager, BaseUrl) ClientM a }
  deriving (Functor, Applicative, Monad)


runGymClient :: Manager -> BaseUrl -> GymClient a -> IO (Either ServantError a)
runGymClient m u client = runExceptT $ runReaderT (getGymClient client) (m, u)

getConnection :: GymClient (Manager, BaseUrl)
getConnection = GymClient ask

-- | So that we don't have to make calls with the manager and baseurl each time
withConnection :: (Manager -> BaseUrl -> ClientM a) -> GymClient a
withConnection fn = do
  (mgr, url) <- getConnection
  GymClient . ReaderT . const $ fn mgr url

-- * Wrapped servant calls

-- | Create an instance of the specified environment
envCreate :: EnvID -> GymClient InstID
envCreate = withConnection . envCreate'

-- | List all environments running on the server
envListAll :: GymClient Environment
envListAll = withConnection envListAll'

-- | Reset the state of the environment and return an initial observation.
envReset :: Text -> GymClient Observation
envReset = withConnection . envReset'

-- | Step though an environment using an action.
envStep :: Text -> Step -> GymClient Outcome
envStep a b = withConnection $ envStep' a b

-- | Get information (name and dimensions/bounds) of the env's @action_space@
envActionSpaceInfo :: Text -> GymClient Info
envActionSpaceInfo = withConnection . envActionSpaceInfo'

envActionSpaceSample :: Text -> GymClient Action
envActionSpaceSample = withConnection . envActionSpaceSample'

envActionSpaceContains :: Text -> Int -> GymClient Object
envActionSpaceContains a b = withConnection $ envActionSpaceContains' a b

-- | Get information (name and dimensions/bounds) of the env's @observation_space@
envObservationSpaceInfo :: Text -> GymClient Info
envObservationSpaceInfo = withConnection . envObservationSpaceInfo'

-- | Start monitoring
envMonitorStart :: Text -> Monitor -> GymClient ()
envMonitorStart a b = withConnection $ envMonitorStart' a b

-- | Flush all monitor data to disk
envMonitorClose :: Text -> GymClient ()
envMonitorClose = withConnection . envMonitorClose'

envClose :: Text -> GymClient ()
envClose = withConnection . envClose'

-- | Flush all monitor data to disk
upload :: Config -> GymClient ()
upload = withConnection . upload'

-- | Request a server shutdown
shutdownServer :: GymClient ()
shutdownServer = withConnection shutdownServer'

