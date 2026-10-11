#!/bin/sh
# Produce a Python-free build tree without modifying the working checkout.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
[ "$#" -eq 1 ] || { echo 'usage: export-source.sh NEW_DIRECTORY' >&2; exit 2; }
[ ! -e "$1" ] || { echo 'destination already exists' >&2; exit 1; }
[ -f "$root/third_party/llama.cpp/ggml/CMakeLists.txt" ] || { echo 'prepare dependencies first' >&2; exit 1; }
parent=$(CDPATH= cd -- "$(dirname "$1")" && pwd)
destination="$parent/$(basename "$1")"
case "$destination/" in "$root/"*) echo "destination must be outside the source tree" >&2; exit 1;; esac
mkdir "$destination"
rsync -a --exclude=.git --exclude=/build --exclude='*.py' --exclude=__pycache__ \
  --exclude=.DS_Store --exclude='*.log' "$root/" "$destination/"
