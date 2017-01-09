{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE TypeOperators         #-}

module Client
  ( envCreate
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
  , EnvID (..)
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

import           Data.Aeson          (FromJSON, Object (..), ToJSON)
import qualified Data.HashMap.Strict as HM (HashMap (..))
import           Data.Proxy          (Proxy (..))
import qualified Data.Text           as T (Text)
import           GHC.Generics
import           Network.HTTP.Client (Manager, defaultManagerSettings,
                                      newManager)
import           Servant.API
import           Servant.Client      (BaseUrl (..), ClientM, client)
import           Servant.HTML.Lucid  (HTML)

--------------------------------------------------------------------------------
-- | Data types and instances

newtype EnvID = EnvID { env_id :: T.Text }
  deriving Generic

instance ToJSON EnvID

newtype InstID = InstID { instance_id :: T.Text }
  deriving (Eq, Show, Generic)

instance ToJSON InstID
instance FromJSON InstID

newtype Environment = Environment { all_envs :: HM.HashMap T.Text T.Text }
  deriving (Eq, Show, Generic)

instance ToJSON Environment
instance FromJSON Environment

newtype Observation = Observation { observation :: [Double] }
  deriving (Eq, Show, Generic)

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
  deriving (Eq, Show, Generic)

instance ToJSON Action
instance FromJSON Action

data Monitor = Monitor
  { directory      :: !T.Text
  , force          :: !Bool
  , resume         :: !Bool
  , video_callable :: !Bool
  } deriving Generic

instance ToJSON Monitor

data Config = Config
  { training_dir :: !T.Text
  , algorithm_id :: !T.Text
  , api_key      :: !T.Text
  } deriving Generic

instance ToJSON Config

instance MimeUnrender HTML () where
    mimeUnrender _ _ = return ()

--------------------------------------------------------------------------------
-- | The API represented as a type

type GymAPI = "v1" :> "envs" :> ReqBody '[JSON] EnvID :> Post '[JSON] InstID
         :<|> "v1" :> "envs" :> Get '[JSON] Environment
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text
              :> "reset" :> Post '[JSON] Observation
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text
              :> ReqBody '[JSON] Step :> "step" :> Post '[JSON] Outcome
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text
              :> "action_space" :> Get '[JSON] Info
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text
              :> "action_space" :> "sample" :> Get '[JSON] Action
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text
              :> "action_space" :> "contains" :> Capture "x" Int
              :> Get '[JSON] Object
         :<|> "v1" :> "envs"
              :> Capture "instance_od" T.Text :> "observation_space"
              :> Get '[JSON] Info
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text :> ReqBody '[JSON] Monitor
              :> "monitor" :> "start" :> Post '[HTML] ()
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text
              :> "monitor" :> "close" :> Post '[HTML] ()
         :<|> "v1" :> "envs"
              :> Capture "instance_id" T.Text :> "close"
              :> Post '[HTML] ()
         :<|> "v1" :> "upload" :> ReqBody '[JSON] Config :> Post '[HTML] ()
         :<|> "v1" :> "shutdown" :> Post '[HTML] ()

--------------------------------------------------------------------------------
-- | Proxy for the API

gymAPI :: Proxy GymAPI
gymAPI = Proxy

--------------------------------------------------------------------------------
-- | Automatically derived query functions

envCreate               :: EnvID -> Manager -> BaseUrl -> ClientM InstID
envListAll              :: Manager -> BaseUrl -> ClientM Environment
envReset                :: T.Text -> Manager -> BaseUrl -> ClientM Observation
envStep                 :: T.Text -> Step -> Manager -> BaseUrl -> ClientM Outcome
envActionSpaceInfo      :: T.Text -> Manager -> BaseUrl -> ClientM Info
envActionSpaceSample    :: T.Text -> Manager -> BaseUrl -> ClientM Action
envActionSpaceContains  :: T.Text -> Int -> Manager -> BaseUrl -> ClientM Object
envObservationSpaceInfo :: T.Text -> Manager -> BaseUrl -> ClientM Info
envMonitorStart         :: T.Text -> Monitor -> Manager -> BaseUrl -> ClientM ()
envMonitorClose         :: T.Text -> Manager -> BaseUrl -> ClientM ()
envClose                :: T.Text -> Manager -> BaseUrl -> ClientM ()
upload                  :: Config -> Manager -> BaseUrl -> ClientM ()
shutdownServer          :: Manager -> BaseUrl -> ClientM ()

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
