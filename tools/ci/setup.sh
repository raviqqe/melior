#!/bin/sh

set -e

emerge-webrsync
emerge -g sys-apps/grep sys-devel/llvm dev-util/rustup

rustup default stable
