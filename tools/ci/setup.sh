#!/bin/sh

set -e

pacman -Syu --noconfirm base-devel llvm rustup

rustup default stable
