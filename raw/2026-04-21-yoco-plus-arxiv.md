---
title: "YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.13556
collected: 2026-04-21
tags: [kv-cache, compression, cross-layer, attention, architecture, research]
---

# YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference

**arXiv ID**: 2604.13556  
**Submitted**: April 15, 2026  
**Authors**: You Wu, Ziheng Chen, et al.  
**Code**: https://github.com/wuyou2002/YOCO-plus

## Abstract (paraphrased)

YOCO++ builds on the YOCO cross-layer KV compression method by adding weighted residual connections between the KVs of each bottom-half layer and the bottom layer. This increases model capacity while maintaining the same training and inference efficiency as YOCO, achieving state-of-the-art performance among cross-layer KV compression methods at a 50% KV cache compression rate.

## Background: YOCO (You Only Cache Once)

YOCO is a cross-layer KV sharing method. In a standard Transformer, every layer has its own KV cache. YOCO compresses this by sharing the KVs of the middle layer with the top-half layers — each of the top-half layers uses the same KV tensors from the single shared "global" attention layer instead of computing its own. This reduces KV cache memory by approximately 50% (only the bottom-half layers compute full per-layer KV caches).

The limitation of YOCO: quality degrades because the top-half layers lose layer-specific KV information; the single shared middle-layer KV is a lossy representation of what a full per-layer KV would contain.

## YOCO++ Contribution

YOCO++ adds a **weighted residual connection** between the KVs of each bottom-half layer and the bottom (input) layer. Concretely:

- Each bottom-half layer's KV = f(layer input, residual from bottom-layer KV)
- The residual connection is learned (weighted), not a simple skip connection
- This gives the model more expressive capacity at the cost of a slightly more complex computation during forward pass

**Key implementation detail**: to avoid additional cache I/O overhead during decoding, YOCO++ caches the **combined KVs** (after residual blending) rather than caching raw per-layer outputs and computing the blend at inference time. This is a critical engineering choice: the inference path is identical to YOCO in terms of memory access patterns.

## Performance Claims

- **Compression**: 50% KV cache reduction (same as baseline YOCO)
- **Quality**: SOTA among cross-layer KV compression methods at the 50% compression operating point
- **Compared to YOCO**: YOCO++ achieves higher model capacity (better downstream quality) at identical training and inference efficiency
- **Compared to full-KV Transformer**: YOCO++ outperforms the standard Transformer at the same parameter count (the paper claims this, implying the architectural inductive bias helps)

Note: specific perplexity / benchmark numbers (exact PPL on Wikitext, etc.) not captured in this raw source — refer to the full paper.

## Relationship to vLLM and Production Serving

YOCO++ is an **architecture-level** compression method, not a serving-system technique like PagedAttention or TurboQuant. It requires training (or fine-tuning) the model with the YOCO++ architecture rather than applying compression post-hoc to existing models.

Serving implications:
- A model trained with YOCO++ needs ~50% less KV cache GPU memory than a standard Transformer of the same size
- No runtime overhead during decode (caching combined KVs avoids extra computation on the inference path)
- Requires vLLM to support the YOCO++ attention pattern — unlikely to be in vLLM yet (not a standard Transformer attention variant)
- Relevant to the Q2 2026 vLLM roadmap item "KV cache manager rethink" — future architecture-aware KV layout support could enable YOCO-family models

## Relationship to Other KV Compression Methods

| Method | Approach | Compression | Requires retraining? |
|--------|----------|-------------|----------------------|
| FP8 KV cache | Quantize existing KVs | 2× | No |
| TurboQuant | WHT + quantize existing KVs | 2.6–4.9× | No |
| YOCO | Cross-layer KV sharing | ~2× | Yes (architecture change) |
| YOCO++ | Cross-layer KV sharing + residual | ~2× | Yes (architecture change) |
| xKV / DeltaKV | SVD/residual KV approximation | varies | Post-hoc, no retraining |

YOCO++ competes with xKV and DeltaKV in the "structured compression at ~50% rate" regime but requires architectural commitment.
