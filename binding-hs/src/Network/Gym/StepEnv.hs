{-# LANGUAGE DeriveGeneric #-}
module Network.Gym.StepEnv where

import Network.Gym.GymEnv
import Network.Gym.Prelude

data StepReq = StepReq
  { action :: Int
  } deriving (Generic, Show, Eq)

instance ToJSON StepReq

data StepRes = StepRes
  { observation :: Vector (Vector (Vector Double))
  , reward :: Double
  , done :: Bool
  , info :: HashMap Text Value
  } deriving (Generic, Show, Eq)

instance FromJSON StepRes

instance Observable StepRes where
  observation = Network.Gym.StepEnv.observation

stepEnv :: GymURL -> String -> Int -> IO (Maybe StepRes)
stepEnv host instanceId a = do
  r <- post (host <> "/envs/" <> instanceId <> "/step") $ toJSON . StepReq $ a
  return $ preview responseBody r >>= decode

unsafeStepEnv :: GymURL -> String -> Int -> IO StepRes
unsafeStepEnv host instanceId a = do
  stepEnv host instanceId a >>= \case
    Nothing ->
      error $ "stepping into instance " <> instanceId
            <> " with action " <> show a
            <> " failed"
    Just res -> return res

