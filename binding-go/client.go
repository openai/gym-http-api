package gym

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strconv"
)

// InstanceID uniquely identifies a running instance.
type InstanceID string

func (i InstanceID) path() string {
	return "/v1/envs/" + string(i)
}

// Space stores information about an action space or an
// observation space.
type Space struct {
	// Name is the name of the space, such as "Box", "HighLow",
	// or "Discrete".
	Name string `json:"name"`

	// Properties for Box spaces.
	Shape []int     `json:"shape"`
	Low   []float64 `json:"low"`
	High  []float64 `json:"high"`

	// Properties for Discrete spaces.
	N int `json:"n"`

	// Properties for HighLow spaces.
	NumRows int       `json:"num_rows"`
	Matrix  []float64 `json:"matrix"`
}

// A Client interfaces with a Gym HTTP server.
type Client struct {
	remoteURL url.URL
}

// NewClient creates a client with the given base URL.
//
// For example, the base URL might be
//
//     http://localhost:8080
//
// This fails if the baseURL is invalid.
func NewClient(baseURL string) (*Client, error) {
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("create client: %s", err)
	}
	return &Client{remoteURL: *u}, nil
}

// ListAll lists all instantiated environments.
// The result maps between instance IDs and environment
// IDs.
func (c *Client) ListAll() (map[InstanceID]string, error) {
	var resp struct {
		Result map[InstanceID]string `json:"all_envs"`
	}
	if err := c.get("/v1/envs/", &resp); err != nil {
		return nil, fmt.Errorf("list environments: %s", err)
	}
	return resp.Result, nil
}

// Create creates a new instance of an environment.
func (c *Client) Create(envID string) (InstanceID, error) {
	var resp struct {
		Result InstanceID `json:"instance_id"`
	}
	req := map[string]string{"env_id": envID}
	if err := c.post("/v1/envs/", req, &resp); err != nil {
		return "", fmt.Errorf("create environment: %s", err)
	}
	return resp.Result, nil
}

// Reset resets the environment instance.
//
// The resulting observation type may vary.
// For discrete spaces, it is an int.
// For vector spaces, it is a []float64.
func (c *Client) Reset(id InstanceID) (observation interface{}, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("reset environment: %s", err)
		}
	}()
	var resp struct {
		Observation interface{} `json:"observation"`
	}
	if err := c.post(id.path()+"/reset/", struct{}{}, &resp); err != nil {
		return nil, err
	}
	return normalizeSpaceElem(resp.Observation)
}

// Step takes a step in the environment.
//
// The action type may vary.
// For discrete spaces, it should be an int.
// For vector spaces, it should be a []float64 or a
// []float32.
//
// See Reset() for information on the observation type.
func (c *Client) Step(id InstanceID, action interface{}, render bool) (observation interface{},
	reward float64, done bool, info interface{}, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("step environment: %s", err)
		}
	}()
	req := map[string]interface{}{"action": action, "render": render}
	var resp struct {
		Observation interface{} `json:"observation"`
		Reward      float64     `json:"reward"`
		Done        bool        `json:"done"`
		Info        interface{} `json:"info"`
	}
	err = c.post(id.path()+"/step/", req, &resp)
	if err != nil {
		return
	}
	resp.Observation, err = normalizeSpaceElem(resp.Observation)
	if err != nil {
		return
	}
	return resp.Observation, resp.Reward, resp.Done, resp.Info, nil
}

// ActionSpace fetches the action space.
func (c *Client) ActionSpace(id InstanceID) (*Space, error) {
	return c.getSpace(id, "action_space")
}

// ObservationSpace fetches the observation space.
func (c *Client) ObservationSpace(id InstanceID) (*Space, error) {
	return c.getSpace(id, "observation_space")
}

// SampleAction samples an action uniformly.
//
// The action is turned into a Go type just like Reset()
// turns observations into Go types.
func (c *Client) SampleAction(id InstanceID) (interface{}, error) {
	var resp struct {
		Action interface{} `json:"action"`
	}
	if err := c.get(id.path()+"/action_space/sample", &resp); err != nil {
		return nil, fmt.Errorf("sample action: %s", err)
	}
	if obs, err := normalizeSpaceElem(resp.Action); err != nil {
		return nil, fmt.Errorf("sample action: %s", err)
	} else {
		return obs, nil
	}
}

// ContainsAction checks if an action is contained in the
// action space.
//
// Currently, only int action types are supported.
func (c *Client) ContainsAction(id InstanceID, act interface{}) (bool, error) {
	num, ok := act.(int)
	if !ok {
		return false, fmt.Errorf("contains action: unexpected action type %T", act)
	}
	var resp struct {
		Member bool `json:"member"`
	}
	path := id.path() + "/action_space/contains/" + strconv.Itoa(num)
	if err := c.get(path, &resp); err != nil {
		return false, fmt.Errorf("contains action: %s", err)
	}
	return resp.Member, nil
}

// Close closes the environment instance.
func (c *Client) Close(id InstanceID) error {
	if err := c.post(id.path()+"/close/", struct{}{}, nil); err != nil {
		return fmt.Errorf("close environment: %s", err)
	}
	return nil
}

// StartMonitor starts monitoring the environment.
func (c *Client) StartMonitor(id InstanceID, dir string, force, resume, videoCallable bool) error {
	req := map[string]interface{}{
		"directory":      dir,
		"force":          force,
		"resume":         resume,
		"video_callable": videoCallable,
	}
	if err := c.post(id.path()+"/monitor/start/", req, nil); err != nil {
		return fmt.Errorf("start monitor: %s", err)
	}
	return nil
}

// CloseMonitor stops monitoring the environment.
func (c *Client) CloseMonitor(id InstanceID) error {
	if err := c.post(id.path()+"/monitor/close/", struct{}{}, nil); err != nil {
		return fmt.Errorf("close monitor: %s", err)
	}
	return nil
}

// Upload uploads the monitor results from the directory
// to the Gym website.
//
// If apiKey is "", then the "OPENAI_GYM_API_KEY"
// environment variable is used.
func (c *Client) Upload(dir, apiKey, algorithmID string) error {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_GYM_API_KEY")
	}
	data := map[string]string{"training_dir": dir, "api_key": apiKey}
	if algorithmID != "" {
		data["algorithm_id"] = algorithmID
	}
	if err := c.post("/v1/upload/", data, nil); err != nil {
		return fmt.Errorf("upload: %s", err)
	}
	return nil
}

// Shutdown stops the server.
func (c *Client) Shutdown() error {
	if err := c.post("/v1/shutdown/", struct{}{}, nil); err != nil {
		return fmt.Errorf("shutdown: %s", err)
	}
	return nil
}

func (c *Client) getSpace(id InstanceID, name string) (*Space, error) {
	var resp struct {
		Space *Space `json:"info"`
	}
	if err := c.get(id.path()+"/"+name+"/", &resp); err != nil {
		return nil, fmt.Errorf("get space: %s", err)
	}
	return resp.Space, nil
}

// post encodes data as JSON and POSTs it to the path.
// If result is non-nil, the response is parsed as JSON
// into result.
func (c *Client) post(path string, data, result interface{}) error {
	u := c.remoteURL
	u.Path = path
	body, err := json.Marshal(data)
	if err != nil {
		return err
	}
	resp, err := http.Post(u.String(), "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	return processResponse(resp.Body, result)
}

// get requests the URL and decodes the response as JSON
// into result.
func (c *Client) get(path string, result interface{}) error {
	u := c.remoteURL
	u.Path = path
	resp, err := http.Get(u.String())
	if err != nil {
		return err
	}
	return processResponse(resp.Body, result)
}

func processResponse(body io.ReadCloser, result interface{}) error {
	defer body.Close()
	bodyData, err := ioutil.ReadAll(body)
	if err != nil {
		return err
	}
	if err := responseErrorMessage(bodyData); err != nil {
		return err
	}
	if result != nil {
		if err := json.Unmarshal(bodyData, &result); err != nil {
			return err
		}
	}
	return nil
}

func responseErrorMessage(resp []byte) error {
	var obj struct {
		Message string `json:"message"`
	}
	json.Unmarshal(resp, &obj)
	if obj.Message != "" {
		return errors.New(obj.Message)
	}
	return nil
}

func normalizeSpaceElem(obs interface{}) (interface{}, error) {
	if obs == nil {
		return nil, errors.New("unsupported observation: nil")
	}
	switch obs := obs.(type) {
	case float64:
		return int(obs), nil
	case []interface{}:
		if len(obs) == 0 {
			return nil, errors.New("unsupported observation: empty array")
		} else if _, isFloat := obs[0].(float64); isFloat {
			return normalizeOneDimSpace(obs)
		} else {
			return normalizeMultiDimSpace(obs)
		}
	default:
		return nil, fmt.Errorf("unsupported observation: %v", obs)
	}
}

func normalizeOneDimSpace(obs []interface{}) ([]float64, error) {
	res := make([]float64, len(obs))
	for i, x := range obs {
		var isFloat bool
		res[i], isFloat = x.(float64)
		if !isFloat {
			return nil, errors.New("unsupported observation: heterogeneous array")
		}
	}
	return res, nil
}

func normalizeMultiDimSpace(obs []interface{}) (interface{}, error) {
	firstElem, err := normalizeSpaceElem(obs[0])
	if err != nil {
		return nil, err
	}
	elemType := reflect.TypeOf(firstElem)
	sliceType := reflect.SliceOf(elemType)
	slice := reflect.MakeSlice(sliceType, len(obs), len(obs))
	for i, x := range obs {
		obj, err := normalizeSpaceElem(x)
		if err != nil {
			return nil, err
		}
		val := reflect.ValueOf(obj)
		if val.Type() != elemType {
			return nil, errors.New("unsupported observation: heterogeneous array")
		}
		slice.Index(i).Set(val)
	}
	return slice.Interface(), nil
}
