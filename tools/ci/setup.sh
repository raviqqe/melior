#!/bin/sh

set -e

packages='sys-devel/llvm dev-util/rustup'

emerge-webrsync
emerge --autounmask --autounmask-write -g $packages || dispatch-conf
emerge -g $packages

rustup default stable
