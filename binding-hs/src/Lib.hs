{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators     #-}

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

newtype EnvID = EnvID { env_id :: T.Text }
  deriving Generic

instance ToJSON EnvID

type GymAPI = "v1" :> "envs" :> ReqBody '[JSON] EnvID :> Post '[JSON] Object
         :<|> "v1" :> "envs" :> Get '[JSON] Environment

gymAPI :: Proxy GymAPI
gymAPI = Proxy

envCreate :: EnvID -> Manager -> BaseUrl -> ExceptT ServantError IO Object
envListAll :: Manager -> BaseUrl -> ExceptT ServantError IO Environment

envCreate :<|> envListAll = client gymAPI

run :: IO ()
run = do
  let url = BaseUrl Http "localhost" 5000 ""

  manager <- newManager defaultManagerSettings
  inst    <- runExceptT $ envCreate (EnvID "CartPole-v0") manager url
  res     <- runExceptT $ envListAll manager url

  case inst of
    Left err -> print err
    Right ok -> print $ encode ok

  case res of
    Left err  -> putStrLn $ "Error: " ++ show err
    Right env -> print $ encode env
