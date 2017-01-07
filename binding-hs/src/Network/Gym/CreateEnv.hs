module Network.Gym.CreateEnv where

import Network.Gym.GymEnv
import Network.Gym.Prelude

data CreateEnvRes = CreateEnvRes
  { instanceId :: Text
  } deriving (Show, Eq)

instance FromJSON CreateEnvRes where
  parseJSON (Object v) = CreateEnvRes <$> v .: "instance_id"
  parseJSON _          = mempty

createEnv :: GymURL -> GymEnv -> IO (Maybe CreateEnvRes)
createEnv host envId = do
  r <- post (host <> "/envs") (object [("env_id",  toJSON envId)])
  return $ preview responseBody r >>= decode

unsafeCreateEnv :: GymURL -> GymEnv -> IO CreateEnvRes
unsafeCreateEnv host envId = do
  createEnv host envId >>= \case
    Nothing -> error "Error, environment could not be created"
    Just env -> return env
