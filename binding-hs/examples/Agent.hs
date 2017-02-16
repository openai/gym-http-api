{-# LANGUAGE NamedFieldPuns #-}
module Main where

import Prelude
import Control.Monad (replicateM_, when)

import OpenAI.Gym.Client


setupAgent :: Manager -> BaseUrl -> GymClient InstID
setupAgent manager url = do
  inst@InstID{instance_id} <- envCreate (EnvID CartPoleV0) manager url
  envActionSpaceInfo instance_id manager url
  return inst


withMonitor :: InstID -> Manager -> BaseUrl -> GymClient () -> GymClient Monitor
withMonitor InstID{instance_id} manager url agent = do
  envMonitorStart instance_id configs manager url
  agent
  envMonitorClose instance_id manager url
  return configs
  where
    configs :: Monitor
    configs = Monitor "/tmp/random-agent-results" True False False


exampleAgent :: InstID -> Manager -> BaseUrl -> GymClient ()
exampleAgent InstID{instance_id} manager url = do
  envReset instance_id manager url
  go 0 False
  where
    maxSteps :: Int
    maxSteps = 200

    reward :: Int
    reward = 0

    go :: Int -> Bool -> GymClient ()
    go x done = do
      Action{action} <- envActionSpaceSample instance_id manager url
      Outcome ob reward done _ <- envStep instance_id (Step action True) manager url
      when (not done && x < 200) $ go (x + 1) done



main :: IO ()
main = do
  manager <- newManager defaultManagerSettings
  out <- runExceptT . getGymClient $ do
    inst <- setupAgent manager url
    Monitor{directory} <- withMonitor inst manager url $
      replicateM_ episodeCount $ exampleAgent inst manager url

    -- Upload to the scoreboard.
    -- TODO: Implement environment variable support.
    upload (Config directory "algo" "") manager url

  case out of
   Left err -> print err
   Right ok  -> print $ encode ok

  where
    url :: BaseUrl
    url = BaseUrl Http "localhost" 5000 ""

    episodeCount :: Int
    episodeCount = 100
