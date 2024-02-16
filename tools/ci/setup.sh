#!/bin/sh

set -e

dnf -y install llvm

curl -fsS https://sh.rustup.rs | sh -y
