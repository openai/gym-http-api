{-# LANGUAGE OverloadedStrings #-}

module AppSpec where

import           Control.Exception          (ErrorCall (..), throwIO)
import           Control.Monad.IO.Class     (liftIO)
import           Control.Monad.Trans.Except
import qualified Data.HashMap.Strict        as HM
import           Data.Proxy
import qualified Data.Text                  as T
import           Lib
import           Network.HTTP.Client        (Manager, defaultManagerSettings,
                                             newManager)
import           Network.HTTP.Types
import           Network.Wai.Handler.Warp
import           Servant.API
import           Servant.Client
import           Servant.Mock
import           Servant.Server
import           System.Process             (callCommand)
import           Test.Hspec

type Host = (Manager, BaseUrl)

try :: Host -> (Manager -> BaseUrl -> ExceptT ServantError IO a) -> IO a
try (manager, baseUrl) action = do
  result <- runExceptT $ action manager baseUrl
  case result of
    Right x  -> return x
    Left err -> throwIO $ ErrorCall $ show err

mockServer :: IO Application
mockServer = return $ serve gymAPI $ mock gymAPI Proxy

withServer :: (Host -> IO a) -> IO a
withServer action = testWithApplication mockServer $ \ port -> do
  let url = BaseUrl Http "localhost" port ""
  manager <- newManager defaultManagerSettings
  action (manager, url)

spec :: Spec
spec = do
  describe "OpenAI GYM API" $ around withServer $ do
    context "POST v1/envs/" $ do
      it "returns an instance id" $ \ host -> do
        InstID i <- try host (envCreate (EnvID "CartPole-v0"))
        i `shouldSatisfy` (not . T.null)

    context "GET v1/envs/" $ do
      it "returns a list of environments" $ \ host -> do
        Environment es <- try host envListAll
        es `shouldSatisfy` (not . null)

    context "POST v1/envs/:instance_id/reset/" $ do
      it "returns an observation space" $ \ host -> do
        Observation o <- try host (envReset "id")
        o `shouldSatisfy` (not . null)

    context "POST v1/envs/:instance_id/step/" $ do
      it "returns some information about the environment" $ \ host -> do
        Step o r d i <- try host (envStep "id" (Action 0 False))
        o `shouldSatisfy` (not . null)

