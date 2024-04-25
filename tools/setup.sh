#!/bin/sh

set -e

llvm_version=17

brew install llvm@$llvm_version

echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
pkg-config --libs libzstd
ls -l /opt/homebrew/opt/zstd/lib
