{-# LANGUAGE DeriveGeneric #-}
module Network.Gym.ResetEnv where

import Network.Gym.GymEnv
import Network.Gym.Prelude

data ResetRes = ResetRes
  { observation :: Vector (Vector (Vector Double))
  } deriving (Generic, Show)

instance FromJSON ResetRes

instance Observable ResetRes where
  observation = Network.Gym.ResetEnv.observation

resetEnv :: GymURL -> String -> IO (Maybe ResetRes)
resetEnv host instanceId = do
  r <- post (host <> "/envs/" <> instanceId <> "/reset") Null
  return $ preview responseBody r >>= decode

unsafeResetEnv :: GymURL -> String -> IO ResetRes
unsafeResetEnv host instanceId = do
  resetEnv host instanceId >>= \case
    Nothing -> error "Error, environment could not be reset"
    Just obs -> return obs
