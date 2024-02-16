#!/bin/sh

set -e

emerge-webrsync
emerge -g sys-devel/llvm dev-lang/rust-bin

rustup default stable
