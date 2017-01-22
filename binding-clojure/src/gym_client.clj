(ns gym-client
  (:require [clj-http.client :as client]
            [cheshire.core :refer [parse-string]]))

(def remote-base "http://127.0.0.1:5000")
(def envs-path "/v1/envs/")

(defn post-req
  [route data]
  (-> (str remote-base route)
      (client/post {:form-params data :content-type :json})
      :body
      (parse-string true)))

(defn get-req
  [route]
  (-> (str remote-base route)
      (client/get)
      :body
      (parse-string true)))

(defn env-create
  [env-id]
  (->> {:env_id env-id}
       (post-req envs-path)
       :instance_id))

(defn env-reset
  [instance-id]
  (-> (str envs-path instance-id "/reset/")
      (post-req nil)
      :observation))

(defn env-step
  [instance-id action render]
  (-> (str envs-path instance-id "/step/")
      (post-req {:action action :render render})
      (select-keys [:observation :reward :done :info])))

(defn env-space-info
  "type space - action or observation"
  [instance-id type-space]
  (->> (format "/%s_space/" type-space)
       (str envs-path instance-id)
       (get-req)
       :info))

(defn env-action-space-sample
  [instance-id]
  (-> (str envs-path instance-id "/action_space/sample")
      (get-req)
      :action))

(defn env-action-space-contains
  [instance-id x]
  (-> (str envs-path instance-id "/action_space/contains/" x)
      (get-req)
      :member))

(defn env-monitor-start
  [instance-id dir force resume video]
  (-> (str envs-path instance-id "/monitor/start/")
      (post-req {:directory dir
                 :force force
                 :resume resume
                 :video-callable video})))

(defn env-monitor-close
  [instance-id]
  (-> (str envs-path instance-id "/monitor/close/")
      (post-req nil)))

(defn env-close
  [instance-id]
  (-> (str envs-path instance-id "/close/")
      (post-req nil)))

(defn upload
  [training-dir algorithm-id api-key]
  (-> (str "/v1/upload/")
      (post-req {:training_dir training-dir
                 :algorithm_id algorithm-id
                 :api_key api-key})))

(defn shutdown-server
  []
  (-> (str "/v1/shutdown")
      (post-req nil)))
