-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.Client
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -Wno-orphans #-}
module OpenAI.Gym.Prelude
  ( module P
  , parseSingleton
  , toSingleton
  ) where

import Control.Monad              as P
import Control.Monad.Loops        as P
import Control.Monad.Trans.Except as P (runExceptT)
import Control.Monad.Trans.Reader as P
import Control.Monad.Trans.Class  as P
import Data.Aeson                 as P
import Data.HashMap.Strict        as P (HashMap)
import Data.Proxy                 as P
import Data.Text                  as P (Text)
import GHC.Generics               as P
import Network.HTTP.Client        as P hiding (Proxy, responseBody, responseStatus)
import Servant.API                as P
import Servant.Client             as P
import Servant.HTML.Lucid         as P (HTML)
import Control.Monad.IO.Class     as P
import Prelude                    as P
import Data.Aeson.Types (Parser)

instance MimeUnrender HTML () where
    mimeUnrender _ _ = return ()

parseSingleton :: FromJSON a => (a -> b) -> Text -> Value -> Parser b
parseSingleton fn f (Object v) = fn <$> v .: f
parseSingleton fn f _          = mempty

toSingleton :: ToJSON a => Text -> a -> Value
toSingleton f a = object [ f .= toJSON a ]

