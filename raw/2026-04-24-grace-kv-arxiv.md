---
title: "GRACE: Graph-Guided Adaptive Channel Elimination for KV Cache Compression"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.16983
collected: 2026-04-24
tags: [kv-cache, compression, quantization, channel-elimination, graph, long-context]
---

# Graph-Guided Adaptive Channel Elimination for KV Cache Compression

**arXiv:** 2604.16983  
**Submitted:** April 18, 2026  
**Authors:** Yuanchao Bai et al. — Harbin Institute of Technology (corresponding author: yuanchao.bai@hit.edu.cn)  
**Funding:** National Natural Science Foundation of China (62301188, 92270116, U23B2009), China Postdoctoral Science Foundation (2022M710958), Heilongjiang Postdoctoral Science Foundation (LBH-Z22156)

## Abstract / Core Claim

Channel pruning (eliminating entire KV cache channels/dimensions) has emerged as a KV compression strategy complementary to quantization. Existing methods evaluate channel importance **in isolation**, ignoring inter-channel interactions. GRACE reframes KV cache compression as a graph optimization problem where channels are nodes and edges encode inter-channel dependencies. This graph-aware selection enables more accurate elimination of truly redundant channels.

## Core Algorithm

### Problem Setup
KV cache compression via channel elimination: select which K-dimensional channels to drop from the key and value tensors across all decoder layers and attention heads. The operating condition is the long-context inference serving scenario where KV cache size is the binding constraint.

### Key Contribution: Graph-Based Channel Importance
Rather than scoring each channel independently (as in prior work), GRACE constructs an **inter-channel interaction graph** from attention patterns. Channels with high mutual dependency cannot be safely eliminated independently — one's presence compensates for another's absence. The graph structure identifies groups of channels that must be co-retained or co-eliminated.

**Adaptive elimination**: The graph-guided score determines a threshold for elimination per layer/head, allowing different layers to have different compression rates based on their graph structure.

### Comparison Baseline
GRACE is compared against THINK (a prior channel pruning method). GRACE consistently outperforms THINK across most LongBench subtasks under varying KV budgets.

## Performance Results

- **60% KV cache size reduction** with negligible performance degradation
- Evaluated on LLaMA-3-8B-Instruct and Mistral-7B-Instruct-v0.2
- Benchmark: LongBench (6 diverse task categories for long-context understanding)

## Relationship to Other KV Compression Methods

| Method | What It Eliminates | Basis |
|--------|-------------------|-------|
| FP8 KV (vLLM) | Precision (bits per element) | Hardware support |
| TurboQuant (vLLM) | Precision + distribution rotation | WHT + Lloyd-Max |
| GRACE | Entire channels/dimensions | Graph-guided importance |
| YOCO++ | Entire layers' KV (architectural) | Cross-layer sharing |

GRACE is orthogonal to quantization: one could apply GRACE to reduce dimensions then quantize remaining channels.

## Limitations

- Evaluated only on 7-8B models; scalability to 70B+ unknown
- No benchmark vs quantization baselines (FP8, TurboQuant) — unclear if 60% dimension reduction is more or less memory-efficient than 2-bit quantization for same quality
- No vLLM integration; graph construction overhead uncharacterized

## Deferred Reason

Previously deferred from April 22-24 collect runs due to arXiv HTTP 403. Retrieved via web search snippet on April 24, 2026.
