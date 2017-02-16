module OpenAI.Gym.Client (module X) where

import OpenAI.Gym.Data as X
import OpenAI.Gym.API as X

-- minimal re-exports for dependencies
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


