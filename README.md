# UnbiasedMaya

Private source: https://github.com/robertkarl/UnbiasedMaya

[Review the fork diff on GitHub](https://github.com/robertkarl/UnbiasedMaya/compare/upstream-baseline...main).
`upstream-baseline` is the fixed Project Maya v1.0.34 fork point (`e728e52`).

Project Maya's GLM-5.3-Flash inference engine with native C++ model preparation
and an OpenAI-compatible C++ server. No Python in the supported build,
preparation, or runtime. Initial target: Linux, one NVIDIA GPU, text and tools.

**Validated on RTX A6000 48 GB / guest RAM 36 GiB.** Maya-S runs at about
10–11 tokens/s on the short coding tests below, with experts split across VRAM,
pinned RAM, and NVMe. See [PLAN.md](PLAN.md) for acceptance criteria and
[UPSTREAM.md](UPSTREAM.md) for attribution.

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

The sample experimental unit uses `Restart=no` so failures remain visible and
do not trigger repeated large-model loads. Restart manually after diagnosing a
failure with `systemctl restart unbiased-maya`.

Do not run this and another GPU inference service simultaneously. Keep the
existing service configuration and model archive until Maya has passed API,
function-call, memory, and agent-task checks. Record cold startup and warm
throughput separately. The initial deployment and results are documented below.

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

The native packer prepared the complete three-shard Maya-S model: 1,257 dense/
native tensor records, 42 expert layers, and 943,911,040 bytes in `dense.bin`.
No expert payload was duplicated. Actual text, SSE, function-call/result turns,
and a clean Pi agent task passed.

The current experiment uses 36 GiB configured guest RAM (36864 MiB), requested
after the initial build tests. The sample service starts with a 24 GiB pinned
expert RAM budget and a 32 GiB cgroup limit. This deliberately leaves experts
outside RAM/VRAM; engine tier statistics and guest disk-sector counters will be
recorded together to distinguish logical disk requests from physical NVMe reads.

On Debra, the persistent redundant archive is mounted at `/Volumes/models`.
All three Maya-S shards are archived and SHA-256 verified under
`/Volumes/models/project-maya/Maya-S-v2-IQ2_XXS`. That folder also contains
`SHA256SUMS`, `model-provenance.json`, and `archive-verified.json`.

## A6000 / 36 GiB measurements (2026-10-11)

Model: `Maya-S-v2-IQ2_XXS`, pinned in `models-manifests/`. The VM uses 12 vCPUs
on a Ryzen 7 1700 and a Samsung 970 EVO NVMe volume. Context 8192, greedy
sampling, thinking disabled, no MTP. The engine automatically selected 8 CPU
expert threads, 38.97 GiB of VRAM expert slots, and the explicit 24 GiB RAM tier.
Capacity: 5,880 VRAM slots and 3,623 RAM slots out of 12,096 experts.

| Check | Result |
|---|---|
| 400-token coding/prose decode | 10.33–11.05 tok/s |
| Fresh 3,236-token prompt | 314.95 tok/s; 10.27 s |
| Native function call and result | Correct `add(17,25)` call, then answer 42 |
| Streaming text | Correct content and terminal `[DONE]` |
| Pi task | Wrote exact file, read it back, clean final stop |
| Guest-cold restart to readiness | 55 s after dropping guest page cache |
| Memory | No swap or cgroup OOM; 32 GiB cap includes file cache |

The three 400-token requests read approximately 22–33 GB each from the guest
NVMe device, including their prompts. Engine disk waits were about 38–41 ms
per output token in the two coding cases. This configuration does exercise
the disk tier; the model is not fully resident in VRAM/RAM. Short prompts pay
substantial expert-loading overhead and should not be confused with the larger
prompt throughput above. Tool-result timing excludes the 175 reused tokens.

[Machine-readable results](benchmarks/a6000-36gb.json) include timings and
block-device read deltas. Coding responses were intentionally capped at 400
tokens; this is a runtime/performance check, not a completed-code quality score.
The Pi task was deliberately small, not a Regina suite. Earlier Qwen results
used another model/quant and 48 GiB guest RAM, so comparisons are not isolated
engine benchmarks. HDD archive hashing on the host overlapped these tests; VM
model-copy traffic had ended before measurement.

## Switch back to UnbiasedStrata

The prior source, Qwen model, and service are retained. On this deployment,
stop Maya before starting `unbiased-strata.service`; its service memory cap
was reduced to 32 GiB for the smaller VM. Qwen has not yet been revalidated
at 36 GiB. The older DwarfStar unit is disabled and its NVMe model was removed
after verifying the Debra copy; restore that model and revisit its old 44 GiB
service cap before using it.
