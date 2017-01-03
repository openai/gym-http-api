{-# LANGUAGE DataKinds     #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeOperators #-}

module Lib
    ( run
    ) where

import           Control.Monad.Trans.Except (ExceptT, runExceptT)
import           Data.Aeson
import qualified Data.HashMap.Strict        as HM
import           Data.Proxy
import qualified Data.Text                  as T
import           GHC.Generics
import           Network.HTTP.Client        (Manager, defaultManagerSettings,
                                             newManager)
import           Servant.API
import           Servant.Client

newtype Environment = Environment { all_envs :: Object }
  deriving (Show, Generic)

instance ToJSON Environment
instance FromJSON Environment

type GymAPI = "v1" :> "envs" :> Get '[JSON] Environment

gymAPI :: Proxy GymAPI
gymAPI = Proxy

getEnvs :: Manager -> BaseUrl -> ExceptT ServantError IO Environment
getEnvs = client gymAPI

run :: IO ()
run = do
  manager <- newManager defaultManagerSettings
  res <- runExceptT $ getEnvs manager
       $ BaseUrl Http "localhost" 5000 ""
  case res of
    Left err  -> putStrLn $ "Error: " ++ show err
    Right env -> print env
