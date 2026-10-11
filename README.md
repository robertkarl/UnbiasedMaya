# UnbiasedMaya

Project Maya's GLM-5.3-Flash inference engine with native C++ model preparation
and an OpenAI-compatible C++ server. No Python in the supported build,
preparation, or runtime. Initial target: Linux, one NVIDIA GPU, text and tools.

**Implementation under validation.** See [PLAN.md](PLAN.md) for acceptance
criteria and [UPSTREAM.md](UPSTREAM.md) for source attribution. Performance on
the A6000 48 GB / guest RAM 48 GiB configuration has not yet been measured.

## Prepare on a connected machine

Clone this repository and run `./scripts/prepare-source.sh` once to populate the
pinned llama.cpp source and apply its vocabulary-only adapter. This explicit
staging step downloads source; `make` never does.

Obtain all Maya-S GGUF shards on that connected machine, pin their Hugging Face
revision, and verify published SHA-256 values. Keep a durable archive; copy the
working model to the inference machine's NVMe. The pack must live directly inside
the folder containing all shards.

Copy source including `third_party/llama.cpp` to the offline machine:

```sh
rsync -a --exclude=.git --exclude=/build --exclude='*.py' \
  --exclude=__pycache__ ./ rk@INFERENCE_HOST:UnbiasedMaya/
```

## Build and prepare offline

Prerequisites: CMake 3.24+, C/C++20 compiler, make, and CUDA toolkit. The A6000
uses architecture 86; CUDA 12.4 and GCC 13 are the current validation target.

```sh
cd ~/UnbiasedMaya
CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 make CUDA_ARCHS=86 JOBS=4
make test CXX=g++-13
make pack MODEL=/models/Maya-S/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  PACK=/models/Maya-S/pack
make test-chat VOCAB=/models/Maya-S/pack/tokenizer.gguf
```

The packer refuses to overwrite an existing pack. It writes dense tensors and
indices into a temporary sibling, then atomically publishes the directory.
Experts stay in the original GGUF shards. Failed temporary directories remain
available for diagnosis. Keep external input hash manifests with the model.

## Serve

```sh
build/unbiased-maya \
  --model /models/Maya-S/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  --pack /models/Maya-S/pack --ctx 8192 --host 127.0.0.1 --port 11434
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.
Use model ID `maya-s`. Bind to a LAN address only behind the existing access
controls: this server does not implement API-key authentication. There are no
outbound downloads or remote model calls.

The server supervises a child copy of itself containing the upstream engine.
Requests are serialized. The frontend handles chat templates, reasoning, tools,
and SSE; final tool/content parsing may delay streamed content. Engine timing
and observed time to first token must be reported separately.

Maya selects expert tiers and prompt workspace using available VRAM/RAM. NVMe
serves remaining experts. Explicit environment overrides should be recorded
with benchmarks; they are not universal hardware defaults. In particular,
`STRATA_GLM_RAM_GB` can bound the pinned RAM tier when running under a service
memory limit. Do not force `--glm-ram-resident` unless every remaining expert fits.

`temperature: 0` explicitly selects greedy sampling. Nondefault presence,
frequency, and repetition penalties are rejected because the GLM engine does
not implement them. Vision and external Qwen MTP packs are not supported.
`timings.prompt_per_second` uses freshly processed tokens, excluding cache reuse.

## Operational validation

Do not run this and another GPU inference service simultaneously. Keep the
existing service configuration and model archive until Maya has passed API,
function-call, memory, and agent-task checks. Record cold startup and warm
throughput separately. Initial deployment and measured results will be documented
here after validation; a successful build alone is not a successful deployment.

## Current validation

On the A6000 guest, the CUDA 12.4 / GCC 13 build succeeds from a copied source tree
without Git metadata or Python source files. Native packer regressions pass for
cross-shard offsets, BF16 halfway rounding, trunk/MTP separation, truncated
inputs, and overwrite refusal. Tests using Maya-S's actual tokenizer pass for
multilingual round trips, reasoning, tool-result rendering, argument reordering,
optional arguments preceding required ones, and refusal of incomplete calls.

The frontend requires two narrowly scoped llama.cpp patches, both applied by
`prepare-source.sh`: vocabulary-only GLM5-Next recognition and order-independent
tagged tool arguments. No GLM engine math has been changed.

A full model inference run is still pending the model transfer. Build/test
success is not yet a measured throughput or agent-task result.
