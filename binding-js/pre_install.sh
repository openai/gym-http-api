#!/bin/bash -e

# If gulp does not exist, install it
hash gulp 2>/dev/null || npm i -g gulp
hash tsfmt 2>/dev/null || npm i -g tsfmt