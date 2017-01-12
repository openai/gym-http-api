module Main where

import           OpenAI.Gym.Client

main :: IO ()
main = do
  -- Set up client
  let url = BaseUrl Http "localhost" 5000 ""
  manager <- newManager defaultManagerSettings

  -- Set up environment
  id <- runExceptT $ envCreate (EnvID CartPoleV0) manager url
  case id of
    Left err                           -> print err
    Right ok@InstID {instance_id = id} -> do

      -- Set up agent
      actionSpaceInfo <- runExceptT $ envActionSpaceInfo id manager url

      -- Run experiment with monitor
      let outdir = "/tmp/random-agent-results"
      ms <- runExceptT $ envMonitorStart id (Monitor outdir True False False) manager url

      let episodeCount = 100
          maxSteps     = 200
          reward       = 0
          done         = False

      replicateM_ episodeCount $
        do ob <- runExceptT $ envReset id manager url
           case ob of
             Left err -> print err
             Right _  -> return ()

           let loop x = do
                 act <- runExceptT $ envActionSpaceSample id manager url
                 case act of
                   Left err                  -> print err
                   Right Action {action = a} -> do

                     outcome <- runExceptT $ envStep id (Step a True) manager url
                     case outcome of
                       Left err                         -> print err
                       Right (Outcome ob reward done _) ->
                         when (not done && x < 200) $ loop (x + 1)
           loop 0

      -- Dump result info to disk
      close <- runExceptT $ envMonitorClose id manager url
      case close of
        Left err -> print err
        Right ok -> print $ encode ok

      -- Upload to the scoreboard.
      -- TODO: Implement environment variable support.
      up <- runExceptT $ upload (Config outdir "algo" "") manager url
      case up of
        Left err -> print err
        Right _  -> return ()
