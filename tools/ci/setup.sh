#!/bin/sh

set -e

emerge-webrsync
emerge -g sys-devel/llvm dev-lang/rust

rustup default stable
