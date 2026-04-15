---
title: "Model Runner V2: A Modular and Faster Core for vLLM"
source_url: https://vllm.ai/blog/mrv2
collected: 2026-04-15
published: 2026-03-24
type: blog-post
author: vLLM Team
---

# Model Runner V2: A Modular and Faster Core for vLLM

## Core Design Principles
MRV2 implements three foundational changes:
1. **Modularity** — isolating model-specific logic via a new `ModelState` abstract interface
2. **GPU-native operations** — moving CPU bookkeeping to GPU (Triton kernels)
3. **Async-first design** — treating overlapped CPU/GPU execution as a hard constraint, not a retrofit

## Architecture Changes

### Persistent Batch Redesign
V1 tightly coupled persistent state to per-step model inputs, requiring complex reordering. MRV2 decouples them: maintains a stable state table independent of input layout, uses gather operations to produce correctly ordered inputs each step. Preserves incremental update performance while reducing complexity.

### GPU-Native Input Preparation
Input tensor construction (`input_ids`, `positions`, `query_start_loc`, `seq_lens`) now runs on GPU via Triton kernels. Reduces CPU overhead, lowers code complexity, enables direct consumption of device-side results without synchronization.

### Async-First Architecture
Target: "zero synchronization between CPU and GPU across all supported model and feature combinations."
- GPU-side preparation kernels directly consume rejection sampling results
- Outputs transfer asynchronously in separate CUDA streams
- Fully decoupled from computation

### Modularization via ModelState
New abstract interface encapsulates model-specific logic (multimodal embeddings, attention metadata, CUDA graph capture). Common execution path stays focused. Largest file in MRV2 is under 1,300 lines vs. 6,700+ in V1.

## Performance Data

### Throughput
- Qwen3-0.6B on GB200: **56% throughput increase** by offloading input preparation to GPU (25K vs. 16K output tokens/second)

### Speculative Decoding
- **6.3% lower TPOT** on 4×GB200 with GLM-4.7-FP8 and MTP=1, driven by elimination of CPU-GPU synchronization points

## Sampler Improvements
- **Gumbel-Max kernel** — avoids softmax materialization using stateless in-kernel RNG
- **Top-k logprobs** optimization — finds top-k logits first, computes logprobs only for selected candidates
- **Prompt logprobs** efficiency through fine-grained chunking, including within single prompts
- **Indirection support** (`idx_mapping`) inside kernels reduces request state expansion

## Current Limitations (v0.18.0 era)
Unsupported: linear attention models (Qwen3.5, Nemotron 3 Super), speculative decoding methods beyond Eagle/Eagle3/MTP, EPLB, DBO, logits processors, LoRA.

These gaps are being closed in Q2 2026.

## Enabling
```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```
No API changes required.
