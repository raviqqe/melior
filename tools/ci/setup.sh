#!/bin/sh

set -e

emerge-webrsync
emerge --autounmask --autounmask-write -g sys-devel/llvm dev-util/rustup

rustup default stable
