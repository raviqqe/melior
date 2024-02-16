#!/bin/sh

set -e

emerge -g sys-apps/grep sys-devel/llvm dev-util/rustup

rustup default stable
