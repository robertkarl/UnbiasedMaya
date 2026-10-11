# UnbiasedMaya implementation plan

Status: planned; source inspected and cloned, conversion not implemented, Maya-S not downloaded.
Date: 2026-10-11

## Goal

Run Maya-S on VM 220 on Proxmox `.57`, using its RTX A6000 (48 GB VRAM), 48 GiB guest RAM, and NVMe storage. Apply the same approach as Strata → UnbiasedStrata: preserve the inference engine, replace the Python application layer with C++, and make offline deployment a source copy followed by `make`.

The supported build, model preparation, and serving paths must require no Python, pip, virtualenv, model downloads, or network access on the VM. Downloads happen on Debra. This is an implementation plan, not authorization to publish a new GitHub repository.

## Starting point and constraints

- Maya upstream: https://github.com/mw00/project-maya, inspected commit `e728e52d6053bb7f8ede569b323ae4c3c03df27b`.
- Local checkout: `/Users/robertkarl/Code/UnbiasedMaya`.
- Reusable frontend: `/Users/robertkarl/Code/strata-workspace/UnbiasedStrata`, commit `46c46ecfc11d2315512f21f7549b0a7792ccd2ae`.
- Both projects pin llama.cpp to `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Recheck compatibility while integrating; sharing a revision does not establish model/template compatibility.
- Guest: `rk@192.168.50.8`, CUDA 12.4, GCC 13, GPU architecture `sm_86`. Keep its existing WAN firewall restrictions.
- UnbiasedStrata currently serves Qwen on port 11434. Preserve its source, model, service configuration, and benchmark results. Inspect actual service enablement before changing boot behavior.
- `/mnt/ds4` is a 220 GiB NVMe-backed filesystem, nearly full with the two existing model sets. Recheck both guest free space and Proxmox thin-pool physical capacity before writes.
- Debra has access to persistent, redundant SMB storage. Discover its actual mount/share and confirm available capacity before choosing archive paths. Do not put SMB credentials in source, manifests, logs, or Git.
- Leave unrelated guests, the other GPU, and the active Regina work on other machines alone.

## Acceptance criteria

1. An exported source tree, including pinned dependencies, builds on the offline VM with `make CUDA_ARCHS=86 JOBS=4`, without `.git` metadata or Python installed in the build environment.
2. A native preparation tool builds the GLM pack from all Maya-S GGUF shards without copying the entire expert payload or invoking Python.
3. One C++ server executable provides `/health`, `/v1/models`, and OpenAI-compatible `/v1/chat/completions`, including streaming, reasoning output, and tool calls. It may supervise a child copy of itself, as UnbiasedStrata does.
4. Tokenization, chat formatting, special tokens, stop behavior, and tool parsing are validated for GLM rather than inherited as Qwen assumptions.
5. Maya-S loads and produces coherent output on the actual 48/48 VM without OOM, swap dependence, or WAN access.
6. Record cold startup, prompt processing, generation speed, memory use, and NVMe activity. Run a small agent task and grade its artifact separately from whether the agent finished cleanly.
7. Document reproduction and rollback. Do not promise a token rate before measuring this model; report failure or a hardware limit explicitly if one is reached.

## 1. Establish a minimal fork and dependency boundary

- Create a working branch and record upstream provenance and MIT notices in `UPSTREAM.md`.
- Inventory the supported engine path, Python launcher/server, model packer, template/tokenizer logic, and any build-time generators.
- Vendor or pin llama.cpp and all required dependencies on Debra. Remove CMake's implicit FetchContent fallback from the supported build: a missing dependency must fail with a local preparation instruction.
- Add a Makefile with `build`, `pack`, and focused `test` targets. Use explicit CUDA architecture overrides and portable AVX2 CPU settings for the Ryzen 1700.
- Ensure patch application does not require a Git checkout on the destination; apply required patches while preparing the source tree, or use an idempotent non-Git mechanism.
- Remove superseded Python install/serve paths from the supported distribution once replacements pass. Retain upstream history and attribution. Do not hide Python behind a Makefile target.

Deliverable: a reproducible source bundle that configures and compiles with networking disabled.

## 2. Replace GLM model preparation in C++

Upstream `maya.py::pack_step` runs `tools/iq_pack.py --compat-bf16`. The engine loads `index.txt`, `native_experts.txt`, and `dense.bin`; expert tensors remain in the GGUF shards. Upstream locates those shards relative to the pack's parent directory.

- Implement `maya-pack` using pinned GGUF/ggml readers and quantization traits.
- Discover and validate every split shard; reject missing shards, duplicate tensors, unsupported architecture/type combinations, invalid dimensions, and out-of-bounds offsets.
- Reproduce GLM-specific index records, shard references, native expert offsets, alignment, and dense tensor layout.
- Preserve the upstream per-tensor precision rules: native quantized projections where supported, exact float copies where appropriate, and compatible BF16 conversion with the correct rounding where required. Audit router/bias exceptions rather than applying Qwen rules.
- Read vocabulary and chat template directly from the GGUF in the server. Only export tokenizer files if the supported native path actually needs them.
- Write prepared files to a temporary directory and publish them atomically after validation. Record input hashes, pack format/version, source revision, and options.
- Add small deterministic C++ fixtures covering shard boundaries, offsets, supported quant types, BF16 rounding, and malformed inputs. Use existing synthetic GGUF data where appropriate; validate numerical behavior against independent ggml dequantization and documented expected values.

Deliverable: `make pack MODEL=/path/shard-00001-of-00003.gguf PACK=/path/pack`, requiring neither Python nor network access.

## 3. Adapt the UnbiasedStrata C++ frontend

- Reuse its HTTP handling, engine supervision, request serialization, error propagation, backpressure, and watchdog behavior, preserving attribution.
- Replace Qwen-specific CLI validation, two-shard assumptions, fixed model name, n-gram/PLE setup, and mandatory draft-pack arguments.
- Start Maya's existing GLM engine with `--glm-pack`, an explicit context limit, and automatic prefill sizing. Keep upstream inference math unchanged unless a demonstrated bug requires a separate fix.
- Load GLM vocabulary/template through llama.cpp. First verify that its pinned model loader recognizes Maya-S metadata in vocabulary-only mode; if not, add a narrow metadata/tokenizer adapter rather than loading model weights in the frontend or rewriting the GGUF.
- Audit the GLM engine's READY/INFO/T/DONE protocol, errors, cancellation, reuse accounting, and timing fields against the frontend parser.
- Validate reasoning settings, tool schemas, tool-result turns, special-token termination, and truncation. Do not blindly reuse the Qwen parser patch.
- Check upstream single-GPU speculative decoding support. Begin with supported defaults; do not assume Qwen's MTP settings work for GLM on one GPU.
- Focus the first release on Linux, one NVIDIA GPU, text, and tool use. Vision, dashboard parity, Anthropic compatibility, and multi-GPU support are follow-up work rather than prerequisites.

Deliverable: `build/unbiased-maya --model SHARD1 --pack DIR --ctx N --host ADDR --port PORT`, plus frontend protocol and parser regression tests.

## 4. Stage Maya-S and make storage room

- Resolve and pin the Hugging Face revision for `peasantsmith/GLM-5.3-Flash-Maya-GGUF`, folder `Maya-S-v2-IQ2_XXS`, and verify all three shards against published hashes. Upstream reports approximately 96.5 GB; measure actual bytes.
- Use a resumable non-Python downloader on Debra. Store a manifest with revision, filenames, lengths, and hashes; never credentials or signed download URLs.
- Use the persistent redundant SMB share as the durable archive. Use Debra-local staging if needed for download tools or reliable checksumming.
- Make NVMe room by archiving the inactive DwarfStar Qwen model. Verify a complete durable copy before removing its VM copy. Preserve the working UnbiasedStrata model if capacity permits.
- Budget for Maya shards, prepared dense data, temporary files, logs, and filesystem headroom. Check the Proxmox thin pool as well as the guest filesystem; do not assume a deleted guest file immediately releases thin-pool capacity.
- Copy the verified Maya model to guest NVMe and verify destination hashes. Preserve the pack/shard relative layout. Do not serve the active model directly from SMB.

Deliverable: a verified durable model archive and a verified NVMe working copy, with the existing runtime recoverable.

## 5. Build and deploy offline

- Rsync source and prepared dependencies from Debra, excluding build products, credentials, local environments, and unrelated data. The destination must not need Git network access.
- Build with GCC 13 / CUDA 12.4 / `sm_86`, initially four compiler jobs. Run native preparation and focused tests on the guest.
- Keep UnbiasedStrata available through the build and CPU-only checks. Stop it only when Maya needs the GPU.
- Use a separate Maya service/configuration and log directory. Run under `rk`, with controlled memory limits, no swap, and no simultaneous GPU model services. Start with a modest context such as 8K, then increase based on measured memory.
- Inspect automatic VRAM/RAM expert tier sizing, reserved workspace, available guest memory, and any cgroup-awareness gaps before tuning cache sizes. Allow disk-backed experts; do not force an all-resident configuration that exceeds capacity.
- Verify API access from Debra and WAN isolation independently. Change default boot service only after Maya passes validation; keep explicit switchback instructions.

Deliverable: a healthy LAN-only Maya service on the existing hardware.

## 6. Measure correctness and performance

- Confirm model identity, readable output, EOS handling, bounded generation, and stable multi-turn behavior.
- Test native function calling end-to-end: tool declaration → parsed arguments → supplied result → final response. Cover reasoning enabled/disabled where supported and streamed responses.
- Measure cold startup separately from warm requests. For prompt processing, report fresh and reused tokens separately and ensure cache reuse cannot inflate throughput.
- Reuse the earlier short coding prompts and, within the supported context, synthetic prompt lengths. Measure decode and prefill from engine timings, plus wall time and time to first token where meaningful.
- Capture peak host/GPU memory, cgroup OOM counters, GPU utilization, expert cache statistics, and NVMe read traffic. This distinguishes compute, PCIe, and disk bottlenecks.
- Run a small isolated Pi task after API/tool validation. Preserve transcript and test its output independently; a CLI exit code alone is not a clean agent pass.
- Compare measured Maya-S results with the recorded Qwen/DwarfStar and UnbiasedStrata results, clearly labeling different models and quantizations. Do not treat this as an isolated engine comparison or launch a full Regina sweep automatically.

Deliverable: timestamped run artifacts and a concise results table, including limitations and any failed checks.

## 7. Finish the repository and handoff

- Replace the upstream quickstart with the actual Debra download → archive → rsync → make → pack → serve workflow.
- Document model provenance, build prerequisites, supported API behavior, memory settings, disk layout, measurements, and rollback.
- Add a sample service and manifests without secrets or machine-specific credentials. Update homelab hardware/runtime notes while preserving unrelated local edits.
- Review the final diff, run the relevant tests, and scan every commit intended for publication for credentials. Follow the user's instruction to avoid the zat.env workflow.
- Commit the reviewable conversion locally. Publish only to an explicitly selected/authorized destination.

## Execution order and open questions

Do the packer and frontend compatibility investigation first; these determine whether the conversion is sound. Once the SMB location and capacity are known, the model download can overlap the native implementation. Storage migration precedes the VM model copy. GPU cutover follows the build and CPU checks.

Resolve during implementation:

- Exact SMB archive path and available space.
- Maya-S revision, actual size, and metadata/tokenizer compatibility with pinned llama.cpp.
- Exact GLM pack precision and offset requirements, including MTP tensors.
- Actual single-GPU MTP support and useful settings on the A6000.
- Whether upstream tier sizing respects the service memory budget.
- Sustainable context size and throughput on this particular PCIe/NVMe setup.

No installation, model migration, GPU service switch, or publication is performed merely by writing this plan.
