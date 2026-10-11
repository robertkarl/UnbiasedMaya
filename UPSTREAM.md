# Provenance

UnbiasedMaya derives from Project Maya (MIT), https://github.com/mw00/project-maya,
commit `e728e52d6053bb7f8ede569b323ae4c3c03df27b`.
The GLM inference engine is retained. Python setup, model preparation, and HTTP
serving are replaced by the native build and frontend described in README.md.

The C++ frontend derives from UnbiasedStrata (MIT),
https://github.com/robertkarl/UnbiasedStrata,
commit `46c46ecfc11d2315512f21f7549b0a7792ccd2ae`.
Its GLM changes cover launch arguments, tokenizer metadata, sampling validation,
context headroom, and cache-aware prompt timing. Its Qwen parser patch is not used.

llama.cpp/ggml (MIT) is pinned to upstream commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Run `scripts/prepare-source.sh` on a
connected staging host to obtain it. The vocabulary dependency patch is
`patches/llama-vocab-only-glm5.patch`: GLM5-Next vocabulary-only loads use the
existing GLM vocabulary container. Full-model inference for that unknown
architecture remains rejected by llama.cpp. The engine computes GLM inference.
A second patch, `patches/llama-tagged-argument-order.patch`, accepts required
GLM tagged arguments in any order and optional fields between them; required
fields remain required. The server also mirrors upstream Maya’s thinking-off
closing tag and maps OpenAI medium effort to GLM high effort.

Upstream's `ref/upstream-llama.cpp` records a newer upstream reference revision,
`272aad8b984a4470d26f14bea8d958be68ac7c0c`. Those files remain documentation, not
compiled dependencies. Upstream also patches llama.cpp's image encoder for GLM;
vision is outside this fork's initial supported interface.

`native/pack.cpp` ports the GLM portion of upstream `tools/iq_pack.py`, including
its BF16 precision policy and v3 index / v4 shard-offset formats. Expert weights
are not requantized. Native preparation exports a metadata-only tokenizer GGUF
with unchanged architecture, token vocabulary and chat template, and no split
references. Model files are not included in this repository.
