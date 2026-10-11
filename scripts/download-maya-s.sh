#!/bin/sh
# Explicit connected-host action, never invoked by make or the server.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
[ "$#" -eq 1 ] || { echo 'usage: download-maya-s.sh DESTINATION_DIRECTORY' >&2; exit 2; }
mkdir -p "$1"
cd "$1"
revision=f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da
for i in 1 2 3; do
    file=$(printf 'GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-%05d-of-00003.gguf' "$i")
    curl --fail --location --retry 8 --continue-at - --output "$file" \
      "https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/resolve/$revision/Maya-S-v2-IQ2_XXS/$file"
done
if command -v sha256sum >/dev/null; then sha256sum -c "$root/models-manifests/maya-s.sha256"
else shasum -a 256 -c "$root/models-manifests/maya-s.sha256"; fi
cp "$root/models-manifests/maya-s.json" "$root/models-manifests/maya-s.sha256" .
