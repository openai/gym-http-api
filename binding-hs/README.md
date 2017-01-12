# Haskell Binding for the OpenAI gym open-source library

## Building
```
stack setup
stack build
stack exec binding-hs-exe
```

## Checklist
- [x] Implemented query functions
- [x] Added an example agent
- [ ] Added environment variable functionality to obtain the API key
- [ ] Optimization (lagging can be detected while running the example agent)
- [ ] Test suite

## Required HTTP Libraries
- aeson
- http-client
- servant
- servant-client
- servant-lucid
