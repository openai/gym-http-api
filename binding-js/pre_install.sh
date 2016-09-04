#!/bin/bash -e

# If these dependencies do not exist, install them
hash gulp 2>/dev/null || npm i -g gulp
hash tsfmt 2>/dev/null || npm i -g typescript-formatter
hash typings 2>/dev/null || npm i -g typings

(cd src && typings install)
