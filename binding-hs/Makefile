sos_warn:
	@echo "-----------------------------------------------------------------"
	@echo "       ! File watching functionality non-operational !           "
	@echo "                                                                 "
	@echo " Install steeloverseer to automatically run tasks on file change "
	@echo "                                                                 "
	@echo " See https://github.com/schell/steeloverseer                     "
	@echo "-----------------------------------------------------------------"

GHCID_SIZE ?= 8
GHCI=stack ghci

# development
ghcid:
	ghcid --height=$(GHCID_SIZE) --topmost "--command=$(GHCI)"

# linting
hlint:
	if command -v sos > /dev/null; then sos -p 'app/.*\.hs' -p 'src/.*\.hs' \
	-c 'hlint \0'; else $(MAKE) sos_warn; fi

# ctag generation
codex:
	if command -v sos > /dev/null; then sos -p '.*\.hs' \
	-c 'codex update --force'; else $(MAKE) entr_warn; fi

# run server and execute agent
serve:
	cd .. && python gym_http_server.py

example: 
	stack build && stack exec example

