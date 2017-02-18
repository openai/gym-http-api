-------------------------------------------------------------------------------
-- |
-- Module    :  Main
-- License   :  MIT
-- Stability :  experimental
-- Portability: non-portable
--
-- Example of how to build an agent using OpenAI.Gym.Client
-------------------------------------------------------------------------------
{-# LANGUAGE NamedFieldPuns #-}
module Main where

import Prelude
import Control.Monad (replicateM_, when)

import OpenAI.Gym.Client


setupAgent :: GymClient InstID
setupAgent = do
  inst@InstID{instance_id} <- envCreate (EnvID CartPoleV0)
  envActionSpaceInfo instance_id
  return inst


withMonitor :: InstID -> GymClient () -> GymClient Monitor
withMonitor InstID{instance_id} agent = do
  envMonitorStart instance_id configs
  agent
  envMonitorClose instance_id
  return configs
  where
    configs :: Monitor
    configs = Monitor "/tmp/random-agent-results" True False False


exampleAgent :: InstID -> GymClient ()
exampleAgent InstID{instance_id} = do
  envReset instance_id
  go 0 False
  where
    maxSteps :: Int
    maxSteps = 200

    reward :: Int
    reward = 0

    go :: Int -> Bool -> GymClient ()
    go x done = do
      Action{action} <- envActionSpaceSample instance_id
      Outcome ob reward done _ <- envStep instance_id (Step action True)
      when (not done && x < 200) $ go (x + 1) done


main :: IO ()
main = do
  manager <- newManager defaultManagerSettings

  out <- runGymClient manager url $ do
    inst <- setupAgent
    Monitor{directory} <- withMonitor inst $
      replicateM_ episodeCount (exampleAgent inst)

    -- Upload to the scoreboard.
    -- TODO: Implement environment variable support.
    upload (Config directory "algo" "")

  case out of
    Left err -> print err
    Right ok  -> print $ encode ok

  where
    url :: BaseUrl
    url = BaseUrl Http "localhost" 5000 ""

    episodeCount :: Int
    episodeCount = 100
