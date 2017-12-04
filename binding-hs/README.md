# Haskell Binding for the OpenAI gym open-source library

To run the example agent:

```
stack build && stack exec example
```

This library provides a servant-based REST client to the gym open-source library.
[openai/gym-http-api][openai] itself provides a [python-based REST server][flask]
to the gym open-source library, allowing development in languages other than python.

[openai]:https://github.com/openai/gym-http-api
[flask]:https://github.com/openai/gym-http-api/blob/master/gym_http_server.py

