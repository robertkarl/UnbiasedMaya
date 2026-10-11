#!/bin/sh
# Run only on the connected staging host. Build/runtime never call this script.
set -eu
cd "$(dirname "$0")/.."
pin=3cf03257f219afbe7334045ff7c6a06ac68c627d
if [ -e third_party/llama.cpp ]; then
    echo 'third_party/llama.cpp already exists; refusing to replace it' >&2
    exit 1
fi
tmp=$(mktemp -d)
trap 'rmdir "$tmp" 2>/dev/null || true' EXIT
curl --fail --location --retry 5 "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$pin" -o "$tmp/llama.tar.gz"
tar -xzf "$tmp/llama.tar.gz" -C "$tmp"
patch -d "$tmp/llama.cpp-$pin" -p1 < patches/llama-vocab-only-glm5.patch
patch -d "$tmp/llama.cpp-$pin" -p1 < patches/llama-tagged-argument-order.patch
mv "$tmp/llama.cpp-$pin" third_party/llama.cpp
printf '%s\n' "$pin" > third_party/llama.cpp/UNBIASED_PIN
rm "$tmp/llama.tar.gz"
