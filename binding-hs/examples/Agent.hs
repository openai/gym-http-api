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
{-# LANGUAGE CPP #-}
module Main where

import           Control.Exception.Base
import           Control.Monad          (replicateM_, when)
import           Control.Monad.Catch
import           Prelude

import           Network.HTTP.Client
import           OpenAI.Gym
import           Servant.Client


main :: IO ()
main = do
  manager <- newManager defaultManagerSettings
#if MIN_VERSION_servant_client(0,13,0)
  out <- runClientM example (ClientEnv manager url Nothing)
#else
  out <- runClientM example (ClientEnv manager url)
#endif
  case out of
    Left err -> print err
    Right ok -> print ok

  where
    url :: BaseUrl
    url = BaseUrl Http "localhost" 5000 ""


example :: ClientM ()
example = do
  inst <- envCreate CartPoleV0
  Monitor{directory} <- withMonitor inst $
    replicateM_ episodeCount (agent inst)

  -- Upload to the scoreboard.
  -- TODO: Implement environment variable support.
  upload (Config directory "algo" "")

  where
    episodeCount :: Int
    episodeCount = 100


agent :: InstID -> ClientM ()
agent inst = do
  envReset inst
  go 0 False
  where
    maxSteps :: Int
    maxSteps = 200

    reward :: Int
    reward = 0

    go :: Int -> Bool -> ClientM ()
    go x done = do
      Action a <- envActionSpaceSample inst
      Outcome ob reward done _ <- envStep inst (Step a True)
      when (not done && x < 200) $ go (x + 1) done


withMonitor :: InstID -> ClientM () -> ClientM Monitor
withMonitor inst agent = do
  envMonitorStart inst configs
  agent
  envMonitorClose inst
  return configs
  where
    configs :: Monitor
    configs = Monitor "/tmp/random-agent-results" True False False



