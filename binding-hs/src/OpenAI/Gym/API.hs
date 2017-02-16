-------------------------------------------------------------------------------
-- |
-- Module    :  OpenAI.Gym.API
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveFunctor #-}
module OpenAI.Gym.API where

import OpenAI.Gym.Prelude
import OpenAI.Gym.Data

type GymAPI = "v1" :> "envs" :> ReqBody '[JSON] EnvID :> Post '[JSON] InstID
         :<|> "v1" :> "envs" :> Get '[JSON] Environment
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "reset" :> Post '[JSON] Observation
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> ReqBody '[JSON] Step :> "step" :> Post '[JSON] Outcome
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "action_space" :> Get '[JSON] Info
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "action_space" :> "sample" :> Get '[JSON] Action
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "action_space" :> "contains" :> Capture "x" Int
              :> Get '[JSON] Object
         :<|> "v1" :> "envs"
              :> Capture "instance_od" Text :> "observation_space"
              :> Get '[JSON] Info
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text :> ReqBody '[JSON] Monitor
              :> "monitor" :> "start" :> Post '[HTML] ()
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text
              :> "monitor" :> "close" :> Post '[HTML] ()
         :<|> "v1" :> "envs"
              :> Capture "instance_id" Text :> "close"
              :> Post '[HTML] ()
         :<|> "v1" :> "upload" :> ReqBody '[JSON] Config :> Post '[HTML] ()
         :<|> "v1" :> "shutdown" :> Post '[HTML] ()


gymAPI :: Proxy GymAPI
gymAPI = Proxy


newtype GymClient a =
  GymClient { getGymClient :: ReaderT (Manager, BaseUrl) ClientM a }
  deriving (Functor, Applicative, Monad)


runGymClient :: Manager -> BaseUrl -> GymClient a -> IO (Either ServantError a)
runGymClient m u client = runExceptT $ runReaderT (getGymClient client) (m, u)


envCreate'               :: EnvID   -> Manager -> BaseUrl -> ClientM InstID
envListAll'              :: Manager -> BaseUrl -> ClientM Environment
envReset'                :: Text    -> Manager -> BaseUrl -> ClientM Observation
envStep'                 :: Text    -> Step    -> Manager -> BaseUrl -> ClientM Outcome
envActionSpaceInfo'      :: Text    -> Manager -> BaseUrl -> ClientM Info
envActionSpaceSample'    :: Text    -> Manager -> BaseUrl -> ClientM Action
envActionSpaceContains'  :: Text    -> Int     -> Manager -> BaseUrl -> ClientM Object
envObservationSpaceInfo' :: Text    -> Manager -> BaseUrl -> ClientM Info
envMonitorStart'         :: Text    -> Monitor -> Manager -> BaseUrl -> ClientM ()
envMonitorClose'         :: Text    -> Manager -> BaseUrl -> ClientM ()
envClose'                :: Text    -> Manager -> BaseUrl -> ClientM ()
upload'                  :: Config  -> Manager -> BaseUrl -> ClientM ()
shutdownServer'          :: Manager -> BaseUrl -> ClientM ()


envCreate'
  :<|> envListAll'
  :<|> envReset'
  :<|> envStep'
  :<|> envActionSpaceInfo'
  :<|> envActionSpaceSample'
  :<|> envActionSpaceContains'
  :<|> envObservationSpaceInfo'
  :<|> envMonitorStart'
  :<|> envMonitorClose'
  :<|> envClose'
  :<|> upload'
  :<|> shutdownServer'
  = client gymAPI


getConnection :: GymClient (Manager, BaseUrl)
getConnection = GymClient ask

withConnection :: (Manager -> BaseUrl -> ClientM a) -> GymClient a
withConnection fn = do
  (mgr, url) <- getConnection
  GymClient . ReaderT . const $ fn mgr url

envCreate :: EnvID -> GymClient InstID
envCreate = withConnection . envCreate'

envListAll :: GymClient Environment
envListAll = withConnection envListAll'

envReset :: Text -> GymClient Observation
envReset = withConnection . envReset'

envStep :: Text -> Step -> GymClient Outcome
envStep a b = withConnection $ envStep' a b

envActionSpaceInfo :: Text -> GymClient Info
envActionSpaceInfo = withConnection . envActionSpaceInfo'

envActionSpaceSample :: Text -> GymClient Action
envActionSpaceSample = withConnection . envActionSpaceSample'

envActionSpaceContains :: Text -> Int -> GymClient Object
envActionSpaceContains a b = withConnection $ envActionSpaceContains' a b

envObservationSpaceInfo :: Text -> GymClient Info
envObservationSpaceInfo = withConnection . envObservationSpaceInfo'

envMonitorStart :: Text -> Monitor -> GymClient ()
envMonitorStart a b = withConnection $ envMonitorStart' a b

envMonitorClose :: Text -> GymClient ()
envMonitorClose = withConnection . envMonitorClose'

envClose :: Text -> GymClient ()
envClose = withConnection . envClose'

upload :: Config -> GymClient ()
upload = withConnection . upload'

shutdownServer :: GymClient ()
shutdownServer = withConnection shutdownServer'

