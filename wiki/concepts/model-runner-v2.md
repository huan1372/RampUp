---
title: "Model Runner V2 (MRV2)"
tags: [vllm-core, architecture, execution, compile-time, torch-compile, ray, aot, mla, attention-backend]
created: 2026-04-14
updated: 2026-05-02
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md, raw/2026-04-15-model-runner-v2-blog.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-23-vllm-prs-apr22-23.md, raw/2026-04-24-vllm-v020-release.md, raw/2026-04-28-vllm-prs-apr27-28.md, raw/2026-05-02-vllm-prs-may2.md]
related: [concepts/paged-attention.md, concepts/deepseek-v4-attention.md, techniques/speculative-decoding.md, techniques/tensor-parallelism.md]
---

# Model Runner V2 (MRV2)

## Summary
Model Runner V2 is a ground-up rewrite of vLLM's model execution layer, shipped in vLLM v0.18.0 (opt-in) and made the default in v0.19.0 (April 2026). It replaces V1 with a modular, GPU-native, async-first architecture. The largest source file shrank from 6,700+ lines (V1) to under 1,300 lines (MRV2). (source: raw/2026-04-15-model-runner-v2-blog.md)

## Core Design Principles
Three foundational changes drive MRV2:
1. **Modularity** — model-specific logic isolated in a `ModelState` abstract interface; common execution path stays generic
2. **GPU-native input preparation** — `input_ids`, `positions`, `query_start_loc`, `seq_lens` now built on GPU via Triton kernels; eliminates CPU-GPU data transfers for input construction
3. **Async-first** — target is zero CPU-GPU synchronization across all supported features; output transfers run in separate CUDA streams decoupled from computation

## Architecture Details

### Persistent Batch Redesign
V1 tightly coupled persistent state to per-step model inputs, requiring costly reordering. MRV2 maintains a stable state table independent of input layout and uses gather operations to produce correctly ordered inputs each step. (source: raw/2026-04-15-model-runner-v2-blog.md)

### Sampler Improvements
- **Gumbel-Max kernel** — avoids softmax materialization using stateless in-kernel RNG
- **Top-k logprobs** — computes logprobs only for selected top-k candidates, not all vocab
- **Prompt logprobs** — fine-grained chunking including within a single prompt
- **Indirection support** (`idx_mapping`) inside kernels reduces state expansion

### Feature Coverage (v0.19.0)
- Piecewise CUDA graphs for pipeline parallelism (PR #35162)
- ViT full CUDA graph capture (PR #35963)
- Spec decode rejection sampler with greedy/logprobs (PRs #37238, #37237)
- Multi-modal embeddings for spec decode (PR #36097)
- EPLB (Expert-Parallel Load Balancing) (PR #37488)
- Streaming inputs (PR #37028)

## Performance Data
- **Throughput**: Qwen3-0.6B on GB200 shows **56% throughput increase** from GPU-native input prep alone (25K vs. 16K tokens/s) (source: raw/2026-04-15-model-runner-v2-blog.md)
- **Spec decode**: **6.3% lower TPOT** on 4×GB200 with GLM-4.7-FP8 + MTP=1, from eliminating CPU-GPU sync points
- **Overall**: 1.7× throughput over V0 engine when combined with async scheduling (source: raw/2026-04-15-vllm-v019-release.md)

## Compilation: FX Graph Deserialization Elimination (PR #40151, April 23, 2026)

vLLM uses `torch.compile` with FX graph capture for optimized model execution. PR #40151 (builds on PR #38657) eliminates FX graph deserialization overhead during warm compilation by using the Python execution code directly as the runtime source of truth. Attention submodules are inlined as Python functions rather than stored and deserialized from serialized graph representations.

**Warm compile time improvements:**

| Model | Before (s) | After (s) | Reduction |
|-------|-----------|----------|-----------|
| DeepSeek-V3.2 | 6.05 | 0.27 | −95.5% |
| GLM-4.7-FP8 | 7.07 | 0.46 | −93.5% |
| GPT-OSS-120B | 1.57 | 0.19 | −87.9% |
| Llama-3.3-70B | 3.95 | 0.20 | −94.9% |
| Qwen3.5-35B | 2.91 | 1.36 | −53.3% |

Result: sub-2-second warm compile times for most production models. Directly improves restart latency in production serving environments.

**Distinction warm vs cold compile:** Warm compile = reusing a previously-compiled model graph after process restart. Cold compile = first-time graph capture (still takes longer). This PR only addresses the warm path.

(source: raw/2026-04-23-vllm-prs-apr22-23.md)

## v0.20.0 Advances (April 23, 2026)

### RayExecutorV2
A new `RayExecutorV2` distributed execution backend was added. This replaces or extends the previous Ray-based executor path with cleaner distributed state management, aligned with MRV2's modular design.

(source: raw/2026-04-24-vllm-v020-release.md)

### AOT Compile with Batch-Invariance Mode
`torch.compile` AOT (Ahead-Of-Time) mode now supports **batch-invariance**: compile artifacts computed for one batch size can be reused across different batch sizes. This eliminates redundant recompilation when the batch size changes during serving (common in continuous batching). The Inductor cache is now nested under the AOT directory, providing a unified compile artifact hierarchy.

(source: raw/2026-04-24-vllm-v020-release.md)

### FX Graph Splitting via Codegen
FX graphs can now be split via codegen, enabling more granular graph partitioning for heterogeneous execution. Complements PR #40151 (FX graph deserialization elimination) from April 23.

(source: raw/2026-04-24-vllm-v020-release.md)

### Opaque Objects on torch 2.11
torch 2.11 (the new default) enables Opaque Objects support in torch.compile, allowing non-tensor Python objects to pass through compile boundaries without triggering graph breaks. This is relevant for vLLM's model state objects.

(source: raw/2026-04-24-vllm-v020-release.md)

### Auto-Resolution of CUDAGraph Modes
Model Runner V2 now auto-resolves CUDAGraph modes from the attention backend — previously required manual configuration. Simplifies deployment and reduces misconfiguration errors.

(source: raw/2026-04-24-vllm-v020-release.md)

## Current Status
MRV2 is the **default** execution path as of v0.19.0. V1 remains for unsupported cases. Known V2 gaps (as of v0.18.0): linear attention models (Qwen3.5, Nemotron 3 Super), non-Eagle/MTP spec decode methods, EPLB, DBO, logits processors, LoRA — being closed in Q2 2026.

Enable on v0.18.x: `export VLLM_USE_V2_MODEL_RUNNER=1`. No API changes.

## Relationship to Other Concepts
- Executes models using [PagedAttention](paged-attention.md) for memory management
- Integrates with [Speculative Decoding](../techniques/speculative-decoding.md) via the rejection sampler
- Supports [Tensor Parallelism](../techniques/tensor-parallelism.md) via piecewise CUDA graphs

## Eagle Prefill Metadata Optimization (PR #40410, April 27, 2026)

PR #40410 eliminates redundant attention metadata reconstruction during Eagle speculative decoding in MRV2. Previously, metadata was rebuilt three times per speculative step (target model, Eagle prefill, draft decode). Now it is rebuilt once.

**Mechanism:** A new `CapturedAttentionState` named tuple (in `cudagraph_utils.py`) bundles attention metadata + slot mappings into a single object. `PrefillEagleCudaGraphManager` now accepts a pre-constructed `CapturedAttentionState` from the target model rather than rebuilding; `DecodeEagleCudaGraphManager` continues to build independently (decode has different requirements).

**Performance:** ~5–10% end-to-end latency improvement for Eagle speculative decoding, from eliminating per-step GPU memory operations and kernel invocations for metadata construction.

This complements the earlier v0.20.0 change that added full CUDA graph capture for Eagle prefill (PR enabling CUDA graph Eagle prefill, April 23, 2026): that change captured the graph; this change removes waste within the captured execution.

(source: raw/2026-04-28-vllm-prs-apr27-28.md)

## MLA Prefill Backend Selection (PR #32623, May 1, 2026)

PR #32623 adds `--attention-config.mla_prefill_backend` as a new runtime flag for selecting the MLA prefill backend, extending MRV2's pluggable backend philosophy to the MLA prefill path (previously MLA decode backend was selectable; now prefill is too).

**Default changed:** CUTLASS MLA → FlashInfer MLA. Old flags (`use_cudnn_prefill`, `use_trtllm_ragged_deepseek_prefill`, `disable_flashinfer_prefill`) deprecated with backward-compatible warnings.

**cuDNN eliminated:** cuDNN MLA prefill backend removed — it was not used in production and added maintenance burden.

**Architectural note:** Backend-specific code is now in isolated files, consistent with the MRV2 modularity goal (the original MRV2 design had backend isolation as a first-class requirement; the MLA prefill path had remained monolithic until this PR).

For full details including backend options and performance notes, see [DeepSeek V4 Hybrid Attention](deepseek-v4-attention.md).

(source: raw/2026-05-02-vllm-prs-may2.md)

## Open Questions
- Which models still require MRV1 as of v0.20.0? When will they migrate?
- Cold compile times remain unaddressed by PR #40151 — what is the cold compile time for DeepSeek-V3.2 and does it remain a production concern?
- When does EPLB support land in MRV2?
- Does PR #40151's FX graph inlining affect model correctness for any edge cases (non-standard attention variants, LoRA, etc.)?
- What speedup does AOT batch-invariance mode provide in practice (vs per-batch-size recompilation)?
- How does RayExecutorV2 differ from the previous Ray executor in terms of state management and failure recovery?
- Does FX graph splitting via codegen enable MRV2 to run on heterogeneous GPU clusters (different SM versions in the same fleet)?
- Does changing the MLA prefill default from CUTLASS to FlashInfer (PR #32623) affect MRV2 compile graph size or trace correctness?
