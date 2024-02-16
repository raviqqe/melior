#!/bin/sh

set -e

emerge -g llvm rustup

rustup default stable
