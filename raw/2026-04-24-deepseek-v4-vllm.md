---
title: "DeepSeek V4 + vLLM Day-0 Support (April 24, 2026)"
source_type: blog-post+model-release
source_url: https://vllm.ai/blog/deepseek-v4
collected: 2026-04-24
tags: [deepseek, long-context, sparse-attention, kv-cache, moe, vllm, architecture]
---

# DeepSeek V4 + vLLM Day-0 Support

**Published:** April 24, 2026  
**Sources:** vLLM blog (https://vllm.ai/blog/deepseek-v4), HuggingFace model cards (deepseek-ai/DeepSeek-V4-Pro, deepseek-ai/DeepSeek-V4-Flash), community technical analyses.

## Models Released

| Model | Total Params | Active Params | Architecture |
|-------|-------------|---------------|--------------|
| DeepSeek-V4-Pro | 1.6T | 49B | MoE |
| DeepSeek-V4-Flash | 284B | 13B | MoE |

Both models:
- 1 million token context length
- Apache 2.0 license (open weights)
- Available on HuggingFace and via DeepSeek API
- No CUDA dependency — trained and deployable on Huawei Ascend chips (complete Chinese AI stack)

## Architecture: Hybrid Attention Stack

DeepSeek V4 introduces a hybrid attention architecture combining three innovations: Compressed Sparse Attention (CSA), Heavily Compressed Attention (HCA), and Manifold-Constrained Hyper-Connections (mHC).

### Compressed Sparse Attention (CSA)

CSA makes long-context attention tractable by compressing old context tokens:

1. **Compression**: every m tokens in the KV cache are compressed into a single entry
2. **Retrieval**: a learned **Lightning Indexer** scores compressed blocks and selects the top-k most relevant via a learned scoring function
3. **Hybrid access**: selected compressed blocks are concatenated with a sliding window of recent uncompressed tokens for full-attention processing

Key parameter: compression group size m (specific value not disclosed for V4-Pro; smaller than HCA's m').

This is analogous to hierarchical memory: recent tokens (sliding window) = L1 cache, top-k retrieved compressed blocks = L2 cache, remaining compressed tokens = archived.

### Heavily Compressed Attention (HCA)

HCA extends CSA with a much larger compression group:
- Group size m' = **128** tokens compressed into 1 entry (vs CSA's smaller m)
- Applied to a subset of layers where very coarse context representation suffices
- No retrieval indexer — the 1/128 compression ratio is fixed; all compressed entries are attended to

The combination means some layers do fine-grained CSA retrieval, others do coarse HCA compression, depending on the depth/position in the stack.

### Manifold-Constrained Hyper-Connections (mHC)

mHC replaces the standard residual connection `y = x + F(x)` with a structured variant:

1. The residual mapping is constrained to lie on the **Birkhoff polytope** — the manifold of doubly stochastic matrices
2. Birkhoff polytope constraint → spectral norm of the residual map is bounded by ≤ 1
3. Spectral norm ≤ 1 → signal propagation is **non-expansive by construction**, preventing activation explosion across deep stacks (1M-token models are extremely deep in effective computation)

In practice: mHC strengthens the residual connection (conventional hyper-connections allow richer mixing), while the manifold constraint ensures numerical stability at extreme depths/context lengths. The constraint is enforced via a Sinkhorn normalization step during training.

## Efficiency vs. DeepSeek-V3.2

At 1M token context:

| Model | FLOPs per token (vs V3.2) | KV cache (vs V3.2) |
|-------|--------------------------|---------------------|
| DeepSeek-V4-Pro | **27%** | **10%** |
| DeepSeek-V4-Flash | **10%** | **7%** |

These are remarkable reductions. 90% KV cache reduction means V4-Pro's serving cost at 1M context is comparable to V3.2 at ~100K context. V4-Flash achieves 93% KV cache reduction.

## vLLM Day-0 Implementation

vLLM published a first-principles walkthrough of the new attention mechanism alongside day-0 support. Implementation challenges and solutions:

### Hybrid KV Cache
CSA and HCA require different KV storage strategies:
- CSA compressed blocks: stored in compressed form with Lightning Indexer scoring metadata
- HCA compressed blocks: fixed-ratio compressed, stored separately
- Recent tokens (sliding window): standard PagedAttention block allocation
- vLLM's implementation manages these in a unified KV cache layout

### Kernel Fusion
The retrieval step in CSA (Lightning Indexer scoring + top-k selection + concat with sliding window) is performance-critical. vLLM fuses these into a single kernel to avoid materializing large intermediate attention tensors.

### Disaggregated Serving
The large model size (1.6T params for Pro) and 1M context requirement makes disaggregated prefill-decode serving essential. vLLM's KV Connector API is used for cross-node KV transfer for V4 serving.

### Relation to MLA
DeepSeek V4 retains MLA (Multi-head Latent Attention) from V3.2 for the non-CSA/HCA layers. The hybrid stack is: some layers use standard MLA, some use CSA, some use HCA. vLLM handles all three attention types in a single model execution path.

## Significance

1. **First production-grade 1M context open model**: V4-Pro achieves 1M context at 27% of V3.2's per-token FLOPs — previously, 1M context required either hardware-accelerated linear attention (Mamba, RWKV) or impractical compute budgets.

2. **KV cache efficiency breakthrough**: 10% of V3.2's KV cache at 1M context enables serving long-context requests at costs comparable to standard context lengths on prior models.

3. **Hardware independence**: Deployable on Huawei Ascend without CUDA — significant for geopolitics of AI infrastructure.

4. **Architecture signal**: CSA + HCA hybrid suggests the field is moving toward learned sparse attention as the next architectural paradigm after MLA/GQA.

## Open Questions
- What is the specific value of CSA's compression group size m?
- How does the Lightning Indexer selection work — is it a learned score or content-based (dot-product with compressed entry)?
- What is the quality cost of the 128-token HCA compression on long-document QA tasks?
- Does the mHC Sinkhorn normalization add measurable training overhead?
- At what context length does V4-Pro outperform V3.2 on throughput (breakeven point for CSA/HCA overhead vs KV savings)?
- What is the KV transfer volume for a 1M-context V4-Pro request in disaggregated serving?
