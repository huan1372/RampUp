---
title: "Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.15464
collected: 2026-04-24
tags: [tpu, paged-attention, kernel, mbu, mfu, vllm, sglang, decode, prefill]
---

# Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU

**arXiv:** 2604.15464  
**Submitted:** April 16, 2026  
**Authors:** vanbasten23 and collaborators (based on vLLM GitHub PRs #13379, #14846, #14597)

## Abstract / Core Claim

LLM inference kernels and serving systems remain largely GPU-centric. TPU-based inference has no well-established approach for efficiently mapping the dynamic and ragged execution patterns of LLM serving (variable-length sequences, paged KV cache) onto TPU architectures. Ragged Paged Attention (RPA) provides a TPU-native attention kernel that achieves GPU-competitive utilization levels while integrating with existing serving frameworks.

## Key Technical Contributions

### 1. Fine-Grained Tiling
Efficient dynamic slicing over ragged memory: standard TPU matmul tiles assume fixed-size dense tensors, but paged KV cache blocks have non-contiguous layouts with variable per-sequence lengths. RPA uses fine-grained tiling strategies that handle the "ragged" structure (variable sequence lengths and block table indirection) without padding waste.

### 2. Custom Software Pipeline
Fuses KV cache updates with attention computation in a single Pallas/Mosaic kernel. On TPUs, kernel launch overhead is amortized differently than GPUs — RPA pipelines the KV write (from current token) with the attention read (for context tokens) to overlap the two operations.

### 3. Distribution-Aware Compilation
Generates **specialized kernels for three workload modes**:
- **Decode mode**: single-token attention over long context (memory bandwidth-bound)
- **Prefill mode**: full attention over new tokens (compute-bound)
- **Mixed mode**: chunked prefill with concurrent decode (common in vLLM v0.18+)

Each mode has a separately compiled Pallas/Mosaic kernel optimized for its arithmetic intensity.

## Performance Benchmarks

Evaluated on Llama 3 8B on TPU7x:

| Metric | Achieved | What It Means |
|--------|----------|---------------|
| Memory Bandwidth Utilization (MBU) in decode | **86%** | Very close to hardware peak for memory-bound workload |
| Model FLOPs Utilization (MFU) in prefill | **73%** | High fraction of theoretical peak for compute-bound workload |
| Token throughput vs pre-integration baseline | **5× improvement** | Since integration into vLLM-TPU (February 2025) |

## Implementation

RPA is implemented using **Pallas** (Google JAX kernel authoring API) and **Mosaic** (TPU compiler backend). It has been:
- Integrated as the **primary TPU attention backend in vLLM** (PR #13379, February 2025)
- Integrated as the **primary TPU attention backend in SGLang**
- This is the first paper to formally document and benchmark the approach

## Hardware Context

TPU7x is Google's 7th-generation TPU. RPA targets Google Cloud TPU v4/v5 family (exact generation mapping to "TPU7x" not fully specified in available sources).

## Relationship to GPU PagedAttention

RPA is **not a GPU kernel** — it is a TPU-native reimplementation of the PagedAttention concept adapted for:
- TPU memory hierarchy (HBM but no L2 cache)
- Systolic array compute units (matmul-centric, not CUDA-core-centric)
- Pallas/Mosaic compilation model (vs. CUDA PTX)

The block table abstraction from GPU PagedAttention is preserved; only the kernel implementation changes.

## Deferred Reason

Previously deferred from April 23-24 collect runs due to arXiv HTTP 403. Retrieved via web search snippet on April 24, 2026.
