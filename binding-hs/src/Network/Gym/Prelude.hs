module Network.Gym.Prelude
  ( module X
  ) where

import Data.Text           as X (Text)
import Data.HashMap.Strict as X (HashMap)
import Data.Vector         as X (Vector)

import Network.Wreq        as X hiding (Proxy)
import Lens.Micro.Platform as X hiding ((.=), to)

import Data.Monoid  as X
import Data.Aeson   as X
import GHC.Generics as X
import Data.Proxy   as X
