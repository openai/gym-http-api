(ns example-agent
  (:require [boot.core :as boot :refer [deftask]]
            [gym-client :refer :all]))

(defn done-in-loop
  [instance-id]
  (let [step (partial env-step instance-id)]
    (-> (env-action-space-sample instance-id)
        (step true)
        :done)))

(defn run-example
  [env-id output-dir episode-count max-steps
   {:keys [force resume video done render]}]
  (let [instance-id (env-create env-id)]
    (env-monitor-start instance-id output-dir force resume video)
    (doseq [_ (range episode-count)]
      (env-reset instance-id)
      (loop [s max-steps
             done done]
        (when-not done
          (recur (inc s) (done-in-loop instance-id)))))
    (env-monitor-close instance-id)))

(deftask example
  []
  (run-example "Skiing-v0" "tmp" 100 200
               {:force true
                :resume false
                :video true
                :done false
                :render false}))
