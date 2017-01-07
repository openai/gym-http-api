ifndef GHCID_SIZE
  ghcid_size=8
else
  ghcid_size=$(GHCID_SIZE)
endif

# to configure for test or src-specific main files
stack-ghci=stack ghci
stack-ghci-test=stack ghci --test

ghci:
        $(stack-ghci)

ghci-test:
        $(stack-ghci-test)

ghcid:
        ghcid --height=$(ghcid_size) --topmost "--command=$(stack-ghci)"

ghcid-test:
        ghcid --height=$(ghcid_size) --topmost "--command=$(stack-ghci-test)"


