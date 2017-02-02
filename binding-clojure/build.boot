(set-env!
 :resource-paths #{"src"}
 :dependencies '[[me.raynes/conch "0.8.0"]
                 [clj-http "2.3.0"]
                 [cheshire "5.7.0"]])

(require '[example-agent :refer [example]])
