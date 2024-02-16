#!/bin/sh

set -e

dnf -y install llvm rustup

rustup default stable
