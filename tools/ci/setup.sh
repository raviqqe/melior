#!/bin/sh

set -e

emerge-webrsync
emerge --autounmask -g sys-apps/grep sys-devel/llvm dev-util/rustup

rustup default stable
