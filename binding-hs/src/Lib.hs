{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TypeOperators              #-}

module Lib
  ( run
  , envCreate
  , envListAll
  , envReset
  , EnvID (..)
  , InstID (..)
  , Observation (..)
  , Environment (..)
  , gymAPI
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
import           Test.QuickCheck.Arbitrary
import           Test.QuickCheck.Instances

newtype EnvID = EnvID { env_id :: T.Text }
  deriving Generic

instance ToJSON EnvID
instance FromJSON EnvID

newtype InstID = InstID { instance_id :: T.Text }
  deriving (Eq, Show, Arbitrary, Generic)

instance ToJSON InstID
instance FromJSON InstID

newtype Environment = Environment { all_envs :: HM.HashMap T.Text T.Text }
  deriving (Eq, Show, Arbitrary, Generic)

instance ToJSON Environment
instance FromJSON Environment

newtype Observation = Observation { observation :: [Double] }
  deriving (Eq, Show, Arbitrary, Generic)

instance ToJSON Observation
instance FromJSON Observation

type GymAPI =
  "v1" :> "envs" :>
    ( ReqBody '[JSON] EnvID :> Post '[JSON] InstID
 :<|> Get '[JSON] Environment
 :<|> Capture "instance_id" T.Text :> "reset" :> Post '[JSON] Observation
    )

gymAPI :: Proxy GymAPI
gymAPI = Proxy

envCreate :: EnvID -> Manager -> BaseUrl -> ExceptT ServantError IO InstID
envListAll :: Manager -> BaseUrl -> ExceptT ServantError IO Environment
envReset :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO Observation

envCreate :<|> envListAll :<|> envReset = client gymAPI

run :: IO ()
run = do
  let url = BaseUrl Http "localhost" 5000 ""

  manager <- newManager defaultManagerSettings
  inst    <- runExceptT $ envCreate (EnvID "CartPole-v0") manager url
  res     <- runExceptT $ envListAll manager url

  case inst of
    Left err -> print err
    Right ok@InstID {instance_id = id} -> do
      print ok
      obs <- runExceptT $ envReset id manager url
      case obs of
        Left err -> print err
        Right o  -> print o

  case res of
    Left err  -> putStrLn $ "Error: " ++ show err
    Right env -> print env
