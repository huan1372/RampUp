---
title: "vLLM PRs May 1–2, 2026: DeepSeek-V4 Tile Kernels, HMA KV Offload Complete, MLA Prefill Backend Abstraction"
source_type: github-prs
source_url: https://github.com/vllm-project/vllm/pulls
collected: 2026-05-02
tags: [vllm, deepseek, kv-cache, hma, mla, kernel-fusion, performance, attention-backend, tilelang]
---

# vLLM PRs May 1–2, 2026

Collected from vLLM main branch. No new numbered release (v0.20.0 remains current).

---

## PR #41255 — [Perf] DeepSeek-V4 Tile Kernel: head_compute_mix_kernel

**Author:** Isotr0py  
**Merged:** May 1, 2026

Ports `head_compute_mix_kernel` from DeepSeek's TileKernels repository into vLLM. The kernel fuses the head computation mixing step in DeepSeek-V4's multi-head collective (MHC) attention mechanism. Implemented using TileLang (a specialized tile-level kernel DSL) with configurable `h_block` and `n_thr` parameters tuned for Blackwell architecture.

**Benchmarks** (SPEED-Bench, 500 prompts, 4×GB200):

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Request throughput | 4.57 req/s | 4.99 req/s | +9.2% |
| Output tok/s | 1083.05 | 1159.71 | +7.1% |
| Total tok/s | 47,876 | 52,266 | +9.2% |
| Mean TTFT | 51.2 ms | 48.1 ms | −6.1% |

**Scope:** DeepSeek-V4 / V4-Flash MHC attention layers only. Blackwell-optimized (GB200 tested). No change to other attention variants.

---

## PR #41445 — [kv_offload+HMA][13/N]: Enable HMA Support (Final)

**Author:** orozery  
**Merged by:** markmc, May 1, 2026

**This is the final PR in the HMA KV offload series.** The offloading connector now advertises `SupportsHMA`, indicating it can handle Hybrid/Heterogeneous Model Architecture KV caches — defined as models with multiple KV cache groups, sliding window attention, or varying block sizes across groups.

**Series summary** (key milestones):
- PR #38453 (Apr 22): multi-group KV transfer
- PR #39403 (Apr 25): multi-group KV store  
- PR #41228 (May 1): sliding window group support in scheduler
- PR #41445 (May 1): final connector-level enablement; adds unit + e2e tests

**User-confirmed:** Qwen 3.6 B model with `--kv-offloading-size 16` works correctly.

**Significance:** The HMA series enables KV cache offloading for architectures that were previously unsupported: models combining full-attention layers with SWA layers (Mistral-class, Gemma variants), MLA-based models (DeepSeek family with CSA/HCA/mHC KV groups), and any model with heterogeneous block sizes. Prior to this series, KV offloading only worked for simple single-group architectures.

No benchmark numbers captured; this was a correctness/enablement PR completing a prerequisite infrastructure track.

---

## PR #32623 — [Attention] Abstract MLA Prefill Backends and Eliminate cuDNN

**Author:** MatthewBonanni  
**Merged:** May 1, 2026

Refactors MLA (Multi-head Latent Attention) prefill backend selection to use a unified `--attention-config.mla_prefill_backend` flag, mirroring the existing decode backend abstraction. Backend-specific logic moved to separate well-organized files.

**Backends available:**
- `flashinfer` — new default (was CUTLASS MLA)
- `trtllm_ragged_deepseek` — TRT-LLM ragged DeepSeek implementation

**cuDNN eliminated:** cuDNN MLA prefill backend removed for simplicity; it was "generally not used" in practice.

**Deprecated flags** (backward-compatible with deprecation warnings):
- `use_cudnn_prefill`
- `use_trtllm_ragged_deepseek_prefill`
- `disable_flashinfer_prefill`

**Performance impact:** Not benchmarked — described as "TBD" in PR. Default change from CUTLASS MLA to FlashInfer MLA may affect performance in some configurations; no numbers captured.

**Architectural significance:** This enables programmatic backend selection for MLA prefill without recompilation, aligned with the V2 model runner's pluggable backend design. Directly relevant to DeepSeek V4 / V3.2 deployments where MLA prefill is the dominant compute path.

---

## PR #36823 — [vLLM IR] 2/N fused_add_rms_norm and maybe_inplace overload

**Author:** ProExpertProg  
**Merged:** May 2, 2026

Ports `fused_add_rms_norm` into the vLLM IR framework. The op fuses residual add + RMS normalization in a single kernel. Adds `maybe_inplace` overload for in-place memory reuse via functionalization passes.

**Performance:** Benchmark results within noise threshold in E2E serving tests (DeepSeek-V3.1, Qwen3-30B). Eager-path shows ~3.6% regression vs main — dispatcher cost in non-compiled paths. This is an IR plumbing PR, not a direct perf win; enables future optimization passes that can exploit the fused op.

---

## PR #40830 — [MM][CG] Support ViT CG for Qwen2.5-VL

**Author:** johncalesp  
**Merged:** May 2, 2026

Enables CUDA graph capture for the Vision Transformer (ViT) encoder in Qwen2.5-VL, following the same pattern established for Qwen3-VL. Reduces CPU overhead in the vision processing pipeline.

**Benchmarks:**

| Metric | With CG | Without CG | Delta |
|--------|---------|-----------|-------|
| Request throughput | 3.63 req/s | 3.55 req/s | +2.3% |
| Total tok/s | 11,622 | 11,341 | +2.5% |
| Mean TTFT | 36,690 ms | 37,870 ms | −3.1% |

**Scope:** Multimodal ViT encoder only; no impact on text-only inference paths.

---

## Releases

No new vLLM release. v0.20.0 (April 24, 2026) remains current.

## Filtered Out

| PR | Reason |
|----|--------|
| #40796 Gemma 4 soft-token clamp bugfix | Model correctness, no perf impact |
| #41492 Step3Text AutoWeightsLoader refactor | Non-performance refactor |
| #41358 Doc: Codex usage example | Documentation only |
| #41405 ROCm bias dtype cast fix | ROCm-specific minor bugfix |
| #39570 Gemma 4 chat template sync | Template only |
