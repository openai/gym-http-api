#include "include/gym/gym.h"
#include <boost/enable_shared_from_this.hpp>

#include <curl/curl.h>

#include <json/value.h>
#include <json/reader.h>

#include <stdio.h>
#include <random>

namespace Gym {

static bool verbose = false;

static std::random_device rd;
static std::mt19937 rand_generator(rd());

std::vector<float> Space::sample()
{
	if (type==DISCRETE) {
		std::uniform_int_distribution<int> randint(0, discreet_n-1);
		std::vector<float> r(1, 0.0f);
		r[0] = randint(rand_generator);
		return r;
	}

	assert(type==BOX);
	std::uniform_real_distribution<float> rand(0.0f, 1.0f);
	int sz = 1;
	for (int dim: box_shape)
		sz *= dim;
	assert((int)box_high.size()==sz);
	assert((int)box_low.size()==sz);

	std::vector<float> r(sz, 0.0f);
	for (int c=0; c<sz; ++c)
		r[c] = (box_high[c]-box_low[c])*rand(rand_generator) + box_low[c];
	return r;
}

static
std::string require(const Json::Value& v, const std::string& k)
{
	if (!v.isObject() || !v.isMember(k))
		throw std::runtime_error("cannot find required parameter '" + k + "'");
	return v[k].asString();
}

static
boost::shared_ptr<Space> space_from_json(const Json::Value& j)
{
	boost::shared_ptr<Space> r(new Space);
	Json::Value v = j["info"];
	std::string type = require(v, "name");
	if (type=="Discrete") {
		r->type = Space::DISCRETE;
		r->discreet_n = v["n"].asInt(); // will throw runtime_error if cannot be converted to int

	} else if (type=="Box") {
		r->type = Space::BOX;
		Json::Value shape = v["shape"];
		Json::Value low   = v["low"];
		Json::Value high  = v["high"];
		if (!shape.isArray() || !low.isArray() || !high.isArray())
			throw std::runtime_error("cannot parse box space (1)");
		int l1 = low.size();
		int l2 = high.size();
		int ls = shape.size();
		int sz = 1;
		for (int s=0; s<ls; ++s) {
			int e = shape[s].asInt();
			r->box_shape.push_back(e);
			sz *= e;
		}
		if (sz != l1 || l1 != l2)
			throw std::runtime_error("cannot parse box space (2)");
		r->box_low.resize(sz);
		r->box_high.resize(sz);
		for (int i=0; i<sz; ++i) {
			r->box_low[i]  = low[i].asFloat();
			r->box_high[i] = high[i].asFloat();
		}

	} else {
		throw std::runtime_error("unknown space type '" + type + "'");
	}

	return r;
}


// curl

static
std::size_t curl_save_to_string(void* buffer, std::size_t size, std::size_t nmemb, void* userp)
{
	std::string* str = static_cast<std::string*>(userp);
	const std::size_t bytes = nmemb*size;
	str->append(static_cast<char*>(buffer), bytes);
	return bytes;
}

class ClientReal: public Client, public boost::enable_shared_from_this<ClientReal> {
public:
	std::string addr;
	int port;

	boost::shared_ptr<CURL> h;
	boost::shared_ptr<curl_slist> headers;
	std::vector<char> curl_error_buf;

	ClientReal()
	{
		CURL* c = curl_easy_init();
		curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1);
		curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, 3000);
		curl_easy_setopt(c, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
		curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, true);
		curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, 0);
		curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, 0);
		curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, &curl_save_to_string);
		curl_error_buf.assign(CURL_ERROR_SIZE, 0);
		curl_easy_setopt(c, CURLOPT_ERRORBUFFER, curl_error_buf.data());
		h.reset(c, std::ptr_fun(curl_easy_cleanup));
		headers.reset(curl_slist_append(0, "Content-Type: application/json"), std::ptr_fun(curl_slist_free_all));
	}

	Json::Value GET(const std::string& route)
	{
		std::string url = "http://" + addr + route;
		if (verbose) printf("GET %s\n", url.c_str());
		curl_easy_setopt(h.get(), CURLOPT_URL, url.c_str());
		curl_easy_setopt(h.get(), CURLOPT_PORT, port);
		std::string answer;
		curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &answer);
		curl_easy_setopt(h.get(), CURLOPT_POST, 0);
		curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, 0);

		CURLcode r;
		r = curl_easy_perform(h.get());
		if (r) throw std::runtime_error(curl_error_buf.data());

		Json::Value j;
		throw_server_error_or_response_code(answer, j);
		return j;
	}

	Json::Value POST(const std::string& route, const std::string& post_data)
	{
		std::string url = "http://" + addr + route;
		if (verbose) printf("POST %s\n%s\n", url.c_str(), post_data.c_str());
		curl_easy_setopt(h.get(), CURLOPT_URL, url.c_str());
		curl_easy_setopt(h.get(), CURLOPT_PORT, port);
		std::string answer;
		curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &answer);
		curl_easy_setopt(h.get(), CURLOPT_POST, 1);
		curl_easy_setopt(h.get(), CURLOPT_POSTFIELDS, post_data.c_str());
		curl_easy_setopt(h.get(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)post_data.size());
		curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, headers.get());

		CURLcode r = curl_easy_perform(h.get());
		if (r) throw std::runtime_error(curl_error_buf.data());

		Json::Value j;
		throw_server_error_or_response_code(answer, j);
		return j;
	}

	void throw_server_error_or_response_code(const std::string& answer, Json::Value& j)
	{
		long response_code;
		CURLcode r = curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &response_code);
		if (r) throw std::runtime_error(curl_error_buf.data());
		if (verbose) printf("%i\n%s\n", (int)response_code, answer.c_str());

		std::string parse_error;
		Json::Reader jr;
		if (!jr.parse(answer, j, false)) {
			parse_error  = jr.getFormattedErrorMessages();
			parse_error += "original json that caused error: " + answer;
		} else if (!j.isObject()) {
			parse_error  = "top level json is not an object";
			parse_error += "original json that caused error: " + answer;
		}

		if (response_code != 200 && j.isObject() && j.isMember("message")) {
			throw std::runtime_error(j["message"].asString());
		} else if (response_code != 200) {
			throw std::runtime_error("bad HTTP response code, and also cannot parse server message: " + answer);
		} else {
			// 200, but maybe invalid json
			if (!parse_error.empty())
				throw std::runtime_error(parse_error);
		}
	}

	boost::shared_ptr<Environment> make(const std::string& env_id) override;
};

boost::shared_ptr<Client> client_create(const std::string& addr, int port)
{
	boost::shared_ptr<ClientReal> client(new ClientReal);
	client->addr = addr;
	client->port = port;
	return client;
}


// environment

class EnvironmentReal: public Environment {
public:
	std::string instance_id;
	boost::shared_ptr<ClientReal> client;
	boost::shared_ptr<Space> space_act;
	boost::shared_ptr<Space> space_obs;

	boost::shared_ptr<Space> action_space() override
	{
		if (!space_act)
			space_act = space_from_json(client->GET("/v1/envs/" + instance_id + "/action_space"));
		return space_act;
	}

	boost::shared_ptr<Space> observation_space() override
	{
		if (!space_obs)
			space_obs = space_from_json(client->GET("/v1/envs/" + instance_id + "/observation_space"));
		return space_obs;
	}

	void observation_parse(const Json::Value& v, std::vector<float>& save_here)
	{
		if (!v.isArray())
			throw std::runtime_error("cannot parse observation, not an array");
		int s = v.size();
		save_here.resize(s);
		for (int i=0; i<s; ++i)
			save_here[i] = v[i].asFloat();
	}

	void reset(State* save_initial_state_here) override
	{
		Json::Value ans = client->POST("/v1/envs/" + instance_id + "/reset/", "");
		observation_parse(ans["observation"], save_initial_state_here->observation);
	}

	void step(const std::vector<float>& action, bool render, State* save_state_here) override
	{
		Json::Value act_json;
		boost::shared_ptr<Space> aspace = action_space();
		if (aspace->type==Space::DISCRETE) {
			act_json["action"] = (int) action[0];
		} else if (aspace->type==Space::BOX) {
			Json::Value& array = act_json["action"];
			assert(action.size()==aspace->box_low.size()); // really assert, it's a programming error on C++ part
			for (int c=0; c<(int)action.size(); ++c)
				array[c] = action[c];
		} else {
			assert(0);
		}
		act_json["render"] = render;
		Json::Value ans = client->POST("/v1/envs/" + instance_id + "/step/", act_json.toStyledString());
		observation_parse(ans["observation"], save_state_here->observation);
		save_state_here->done = ans["done"].asBool();
		save_state_here->reward = ans["reward"].asFloat();
	}

	void monitor_start(const std::string& directory, bool force, bool resume) override
	{
		Json::Value data;
		data["directory"] = directory;
		data["force"]  = force;
		data["resume"] = resume;
		client->POST("/v1/envs/" + instance_id + "/monitor/start/", data.toStyledString());
	}

	void monitor_stop() override
	{
		client->POST("/v1/envs/" + instance_id + "/monitor/close/", "");
	}
};

boost::shared_ptr<Environment> ClientReal::make(const std::string& env_id)
{
	Json::Value req;
	req["env_id"] = env_id;
	Json::Value ans = POST("/v1/envs/", req.toStyledString());
	std::string instance_id = require(ans, "instance_id");
	if (verbose) printf(" * created %s instance_id=%s\n", env_id.c_str(), instance_id.c_str());
	boost::shared_ptr<EnvironmentReal> env(new EnvironmentReal);
	env->client = shared_from_this();
	env->instance_id = instance_id;
	return env;
}

} // namespace
