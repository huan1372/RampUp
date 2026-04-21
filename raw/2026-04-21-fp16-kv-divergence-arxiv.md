---
title: "The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.15409
collected: 2026-04-21
tags: [kv-cache, fp16, numerical-precision, correctness, inference, research]
---

# The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference

**arXiv ID**: 2604.15409  
**Submitted**: April 2026 (number 15409 in April batch; after YOCO++ at 13556, so approximately April 18–20, 2026)  
**Authors**: Ranjith Chodavarapu, Lei Xu

## Core Claim

KV caching is universally assumed to be numerically equivalent to cache-free recomputation — i.e., serving a token with its KV cache entry should produce the same result as recomputing the KV from scratch. **This assumption is false under standard FP16 precision.**

## Root Cause

During autoregressive decoding, cache-ON and cache-OFF execution paths traverse floating-point operations in different orders. Because FP16 is non-associative (i.e., `(a + b) + c ≠ a + (b + c)` in floating point), the two paths accumulate rounding errors differently. The divergence is **deterministic**, not random: given the same input, cache-ON always produces a different sequence than cache-OFF, and the difference is architecturally predictable.

Mechanism:
1. When a KV tensor is written to cache, it captures a FP16-rounded value
2. At subsequent decode steps, this stored FP16 value is read back and used in attention computation
3. The recomputation path would compute the same value fresh — but intermediate rounding differs due to operation ordering
4. The error accumulates across layers and sequence positions and cannot be corrected by downstream residual stream operations

## Empirical Results

**Models tested**: LLaMA-2-7B, Mistral-7B-v0.3, Gemma-2-2B  
**Benchmark**: GSM8K  
**Sampling strategies**: greedy decoding, temperature sampling (all strategies)

| Finding | Result |
|---------|--------|
| Token divergence rate (cache-ON vs. cache-OFF) | **100%** across all models, all sampling strategies |
| Direction of divergence (which is more accurate?) | Cache-ON higher accuracy in **8 of 9** conditions |
| FP32 divergence reduction vs. FP16 | **8 orders of magnitude** reduction |
| FP32 token flip rate | **0.0%** (zero flips) |

**Conclusion**: The divergence is systematic and beneficial — KV caching in FP16 produces *different but consistently higher accuracy* results than cache-free FP16 recomputation. This is because the KV cache path's specific rounding pattern happens to be more favorable for the downstream computations.

## Architecture-Specific Divergence Patterns

**GQA (Grouped-Query Attention) models** (Mistral, LLaMA-2): sharp divergence concentrated at the **first layer**, where KV sharing across query groups is most sensitive to rounding in the shared KV.

**Gemma-2** (larger head dimension + sliding window attention): **uniform drift accumulation** across all layers, rather than a sharp spike at layer 0. The larger head dimension means each head's dot products accumulate more FP16 errors.

## Implications for the Knowledge Base

### For KV Cache Quantization
Any quantization scheme (FP8, TurboQuant, etc.) adds an additional source of numerical divergence on top of the baseline FP16 divergence. The "FP8 KV is near-zero quality impact" claim is true empirically but should be understood as: FP8 diverges from BF16 by a small additional margin on top of the baseline FP16 divergence that already exists.

### For Correctness Testing
Testing frameworks that compare cache-ON vs. cache-OFF outputs to verify correctness of KV caching implementations will produce 100% false positives under FP16 — every token will differ even with a correctly implemented cache. Correct comparison requires using FP32 or accounting for the systematic FP16 divergence.

### For Reproducibility
Any reproducibility study or benchmark comparing inference runs (one with caching, one without) is confounded by this systematic divergence. Results are not bit-identical even in the absence of bugs.

### For KV Offloading
When KV blocks are offloaded to CPU and brought back, they pass through an additional memory copy that may change FP16 bit patterns depending on the CPU memory layout. The paper's finding suggests this could compound the existing divergence.

## Relationship to Prior Work

- Prior work (KITTY, TurboQuant, etc.) evaluates quality via perplexity or downstream task accuracy — these metrics aggregate over many tokens and are relatively robust to single-token divergences. The 100% token divergence rate found in this paper applies at the individual-token level.
- The paper does NOT claim that FP16 KV caching is broken for production use — it claims that the assumption of numerical equivalence used in many research papers is wrong, and that this matters for correctness testing, reproducibility, and theoretical analysis.
