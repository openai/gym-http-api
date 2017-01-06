{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE DuplicateRecordFields      #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE TypeOperators              #-}

module Lib
  ( run
  , envCreate
  , envListAll
  , envReset
  , envStep
  , EnvID (..)
  , InstID (..)
  , Observation (..)
  , Environment (..)
  , Outcome (..)
  , Step (..)
  , gymAPI
  ) where

import           Control.Arrow              (left)
import           Control.Monad.Trans.Except (ExceptT, runExceptT)
import           Data.Aeson
import qualified Data.ByteString            as BS
import qualified Data.HashMap.Strict        as HM
import           Data.Proxy
import qualified Data.Text                  as T
import qualified Data.Text.Lazy             as TL
import           Data.Text.Lazy.Encoding
import           GHC.Generics
import           Network.HTTP.Client        (Manager, defaultManagerSettings,
                                             newManager)
import           Network.HTTP.Media         ((//), (/:))
import           Servant.API
import           Servant.Client
import           Servant.HTML.Lucid
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

data Step = Step
  { action :: !Int
  , render :: !Bool
  } deriving Generic

instance ToJSON Step
instance FromJSON Step

data Outcome = Outcome
  { observation :: ![Double]
  , reward      :: !Double
  , done        :: !Bool
  , info        :: !Object
  } deriving (Eq, Show, Generic)

instance ToJSON Outcome
instance FromJSON Outcome

newtype Info = Info { info :: Object }
  deriving (Eq, Show, Generic)

instance ToJSON Info
instance FromJSON Info

newtype Action = Action { action :: Int }
  deriving (Eq, Show, Arbitrary, Generic)

instance ToJSON Action
instance FromJSON Action

data Monitor = Monitor
  { directory      :: !T.Text
  , force          :: !Bool
  , resume         :: !Bool
  , video_callable :: !Bool
  } deriving (Eq, Show, Generic)

instance ToJSON Monitor
instance FromJSON Monitor

data Config = Config
  { training_dir :: !T.Text
  , algorithm_id :: !T.Text
  , api_key      :: !T.Text
  } deriving (Eq, Show, Generic)

instance ToJSON Config
instance FromJSON Config

instance MimeUnrender HTML TL.Text where
    mimeUnrender _ = left show . decodeUtf8'

type GymAPI = "v1" :> "envs" :> ReqBody '[JSON] EnvID :> Post '[JSON] InstID
         :<|> "v1" :> "envs" :> Get '[JSON] Environment
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> "reset" :> Post '[JSON] Observation
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> ReqBody '[JSON] Step :> "step" :> Post '[JSON] Outcome
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> "action_space" :> Get '[JSON] Info
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> "action_space" :> "sample" :> Get '[JSON] Action
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> "action_space" :> "contains" :> Capture "x" Int :> Get '[JSON] Object
         :<|> "v1" :> "envs" :> Capture "instance_od" T.Text :> "observation_space" :> Get '[JSON] Info
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> ReqBody '[JSON] Monitor :> "monitor" :> "start" :> Post '[HTML] TL.Text
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> "monitor" :> "close" :> Post '[HTML] TL.Text
         :<|> "v1" :> "envs" :> Capture "instance_id" T.Text :> "close" :> Post '[HTML] TL.Text
         :<|> "v1" :> "upload" :> ReqBody '[JSON] Config :> Post '[HTML] TL.Text
         :<|> "v1" :> "shutdown" :> Post '[HTML] TL.Text

gymAPI :: Proxy GymAPI
gymAPI = Proxy

envCreate :: EnvID -> Manager -> BaseUrl -> ExceptT ServantError IO InstID
envListAll :: Manager -> BaseUrl -> ExceptT ServantError IO Environment
envReset :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO Observation
envStep :: T.Text -> Step -> Manager -> BaseUrl -> ExceptT ServantError IO Outcome
envActionSpaceInfo :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO Info
envActionSpaceSample :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO Action
envActionSpaceContains :: T.Text -> Int -> Manager -> BaseUrl -> ExceptT ServantError IO Object
envObservationSpaceInfo :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO Info
envMonitorStart :: T.Text -> Monitor -> Manager -> BaseUrl -> ExceptT ServantError IO TL.Text
envMonitorClose :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO TL.Text
envClose :: T.Text -> Manager -> BaseUrl -> ExceptT ServantError IO TL.Text
upload :: Config -> Manager -> BaseUrl -> ExceptT ServantError IO TL.Text
shutdownServer :: Manager -> BaseUrl -> ExceptT ServantError IO TL.Text

envCreate :<|> envListAll
          :<|> envReset
          :<|> envStep
          :<|> envActionSpaceInfo
          :<|> envActionSpaceSample
          :<|> envActionSpaceContains
          :<|> envObservationSpaceInfo
          :<|> envMonitorStart
          :<|> envMonitorClose
          :<|> envClose
          :<|> upload
          :<|> shutdownServer
             = client gymAPI

run :: IO ()
run = do
  let url = BaseUrl Http "localhost" 5000 ""
  let action = Step 0 True

  manager <- newManager defaultManagerSettings
  inst    <- runExceptT $ envCreate (EnvID "CartPole-v0") manager url
  res     <- runExceptT $ envListAll manager url

  case inst of
    Left err -> print err
    Right ok@InstID {instance_id = id} -> do
      print $ encode ok
      obs  <- runExceptT $ envReset id manager url
      step <- runExceptT $ envStep id action manager url
      inf  <- runExceptT $ envActionSpaceInfo id manager url
      samp <- runExceptT $ envActionSpaceSample id manager url
      cont <- runExceptT $ envActionSpaceContains id 0 manager url
      oi   <- runExceptT $ envObservationSpaceInfo id manager url
      ms   <- runExceptT $ envMonitorStart id (Monitor "/tmp/api-test" True False False) manager url
      mc   <- runExceptT $ envMonitorClose id manager url
      ec   <- runExceptT $ envClose id manager url

      case obs of
        Left err -> print err
        Right o  -> print $ encode o
      case step of
        Left err   -> print err
        Right step -> print $ encode step
      case inf of
        Left err -> print err
        Right i  -> print $ encode i
      case samp of
        Left err -> print err
        Right sa -> print $ encode sa
      case cont of
        Left err -> print err
        Right c  -> print $ encode c
      case oi of
        Left err -> print err
        Right i  -> print $ encode i
      case ms of
        Left err  -> print err
        Right mon -> print $ encode mon
      case mc of
        Left err  -> print err
        Right mon -> print $ encode mon
      case ec of
        Left err  -> print err
        Right env -> print $ encode env


  case res of
    Left err  -> putStrLn $ "Error: " ++ show err
    Right env -> print $ encode env
