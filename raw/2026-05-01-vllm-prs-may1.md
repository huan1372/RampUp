---
title: "vLLM GitHub PRs — May 1, 2026"
source_type: github-prs
source_url: https://github.com/vllm-project/vllm/pulls
collected: 2026-05-01
tags: [vllm, quantization, tensor-parallelism, kv-offload, fp8, blackwell, hma, torch-compile]
---

# vLLM GitHub PRs — May 1, 2026

Material performance and algorithm PRs merged to vllm-project/vllm on or around May 1, 2026.
No new numbered release (v0.20.0 remains current).

---

## PR #41326 — Faster per-token FP8 group quant packed kernel for Blackwell

**Merged:** May 1, 2026  
**Tags:** quantization, performance, blackwell

**Summary:** Introduces a faster FP8 per-token group quantization kernel specifically optimized for Blackwell (SM100/SM120) GPUs. The "packed" kernel fuses the FP8 quantization step with the subsequent GEMM dispatch, reducing kernel launch overhead and memory traffic vs the prior two-kernel sequence (quantize, then GEMM). Per-token group quantization applies a distinct FP8 scale per token per group of channels — finer-grained than per-tensor FP8, closer to the accuracy of FP16 at the throughput of FP8.

**Scope:** Blackwell-specific optimization. H100/H200 (Hopper SM90) fallback behavior unchanged.

**Relationship to existing work:**
- Complements PR #38877 (MLA + Group FP8 fusion, merged April 22): that PR fused FP8 quant into MLA attention; this PR targets the GEMM path more broadly
- Relevant to the broader FP8 group GEMM effort tracked in vLLM issue #35792

---

## PR #39505 — [compile] Add FlashInfer FP8 async TP fusion and preserve allreduce fusion ordering

**Merged:** May 1, 2026  
**Tags:** torch.compile, tensor-parallelism, fp8, flashinfer, performance

**Summary:** Adds a `torch.compile`-level fusion pass that combines FlashInfer FP8 GEMM operations with async tensor parallelism (TP) allreduce collectives. Also adds logic to preserve allreduce fusion ordering — preventing the compiler from reordering collectives in ways that would break the overlapped async TP communication schedule.

**Technical context:**
- Async TP overlaps communication with computation by launching allreduce ops asynchronously while the next layer's compute begins; the FP8 quantization step must precede the allreduce, and the compiler must be aware of this ordering dependency
- Previously, torch.compile's fusion passes could inadvertently reorder these ops, breaking async TP overlapping or causing incorrect output for FP8 models on TP>1 configurations
- This PR teaches the compiler about the FP8→allreduce ordering invariant

**Scope:** Applies to models using FP8 quantization with TP≥2. Most relevant for Blackwell B200/GB200 multi-GPU deployments where async TP is the default.

**Relationship to prior issue:** GitHub issue #27985 "[Bug]: Async TP pattern matching fails for fp8 models on Blackwell with the default FlashInfer fp8 gemm" — this PR is the resolution.

---

## PR #41228 — [kv_offload+HMA][12/N]: Scheduler-side support for sliding window groups

**Merged:** May 1, 2026  
**Tags:** kv-offload, hma, scheduler, sliding-window-attention

**Summary:** Part 12 of the ongoing HMA (Hybrid Multi-token Attention / Multi-group Attention) KV offload series. Adds scheduler-side support for **sliding window attention (SWA) groups** in the KV offload path.

**Background:** The HMA KV offload series is extending vLLM's KV cache offloader to handle architectures that use multiple KV groups (e.g., DeepSeek V4's CSA+HCA+mHC layout). Prior PRs in this series (#38453 multi-group transfers, #39403 multi-group store) established the core multi-group infrastructure. PR #40946 (April 27) separately fixed the SWA scheduler admission deadlock for SWA models without KV offload.

**This PR extends** the KV offload scheduler's block management to correctly account for sliding window groups, where only the most recent W tokens' KV blocks are relevant and older blocks can be freed — the offload scheduler must track which blocks are within the active window for each SWA group independently.

**Scope:** Correctness prerequisite for using KV offloading with hybrid full + sliding-window attention architectures (Mistral-class SWA, Gemma SWA, custom hybrid designs). No performance benchmark numbers — infrastructure PR.

**Relationship to existing KB entries:**
- Continues HMA KV offload series (KB: PR #39403, part 11, April 25)
- Interacts with SWA deadlock fix (KB: PR #40946, April 27)
- Enables future KV offloading for architectures combining full attention layers + SWA layers in the same model
