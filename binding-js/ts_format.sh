#!/bin/bash -e
tsfmt -r $(find . -name "*.ts*" ! -name "*.d.ts" ! -path "./node_modules/*") || true
