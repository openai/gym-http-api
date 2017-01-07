module Main where

import Test.DocTest

main :: IO ()
main = doctest
  [ "-XBangPatterns"
  , "-XLambdaCase"
  , "-XOverloadedStrings"
  , "-XViewPatterns"
  , "src"
  ]
