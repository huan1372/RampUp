---
title: "PagedAttention"
tags: [memory, kv-cache, vllm-core, attention, tpu, pallas, mosaic]
created: 2026-04-14
updated: 2026-04-24
sources: [raw/vllm-pagedattention-paper.md, raw/vllm-benchmarks-2026.md, raw/2026-04-14-vllm-rampup-recap.md, raw/2026-04-24-ragged-paged-attention-tpu-arxiv.md]
related: [concepts/kv-cache-management.md, techniques/prefix-caching.md, concepts/continuous-batching.md]
---

# PagedAttention

## Summary
PagedAttention is vLLM's core memory management innovation. It applies the operating system concept of virtual memory and paging to KV cache allocation, eliminating the memory waste caused by pre-allocating contiguous memory blocks for each sequence.

## How It Works
Traditional LLM serving pre-allocates a contiguous block of GPU memory for each sequence's KV cache, sized for the maximum possible sequence length. This leads to massive internal fragmentation — most sequences don't reach max length, so the unused memory is wasted.

PagedAttention divides the KV cache into fixed-size **blocks** (similar to OS memory pages). Each sequence gets a **block table** that maps logical blocks to physical blocks in GPU memory. Blocks are allocated on demand as the sequence grows, and freed immediately when the sequence finishes.

Key mechanisms:
- **Block table**: per-sequence mapping from logical to physical blocks
- **On-demand allocation**: blocks allocated as tokens are generated, not pre-reserved
- **Reference counting**: enables memory sharing for parallel sampling — multiple sequences from the same prompt share the same prompt KV blocks via copy-on-write
- **Near-zero waste**: under 4% memory waste vs. up to 60-80% in naive implementations

## Key Parameters
- `block_size` — number of tokens per block (default: 16)
- `gpu_memory_utilization` — fraction of GPU memory for KV cache (default: 0.9)
- `max_num_seqs` — maximum concurrent sequences (batch size)
- `max_num_batched_tokens` — total tokens per forward pass

## Benchmarks & Numbers
- Reduces memory waste to under 4% (source: vLLM paper, ramp-up recap)
- Reduces memory usage by up to 55% for parallel sampling and beam search
- Enables 2-4x higher throughput compared to HuggingFace Transformers baseline
- Baseline waste without PagedAttention: 60-80% (source: ramp-up recap)

## Key Insight: Statistical Multiplexing
Not every request hits its maximum sequence length at the same time. A shared pool
can serve more requests than individual per-request reservations would allow — the
total pool can actually be **smaller than the sum of all worst-case reservations**
(source: raw/2026-04-14-vllm-rampup-recap.md).

The `gpu_memory_utilization` parameter (default 0.9) controls how aggressively you
bet on statistical sharing. Higher values pack more concurrent requests but raise
the risk of [preemption](kv-cache-management.md) when the pool runs out.

## Ragged Paged Attention: TPU Implementation (arXiv 2604.15464, April 2026)

PagedAttention was originally designed for NVIDIA GPUs. Applying it to TPUs requires a different kernel implementation because TPUs use systolic array compute units and a different memory hierarchy (no L2 cache; Pallas/Mosaic compilation model vs CUDA PTX).

**Ragged Paged Attention (RPA)** is a TPU-native kernel that implements the paged attention block table abstraction using Pallas (Google JAX kernel API) and Mosaic (TPU compiler backend).

**Three key techniques:**
1. **Fine-grained tiling**: handles ragged/non-contiguous block table layout without padding waste — standard TPU matmul tiles assume dense tensors, so custom tiling is needed
2. **Custom software pipeline**: fuses KV cache writes (current token) with attention reads (context tokens) in a single kernel
3. **Distribution-aware compilation**: separate compiled kernels for decode (memory-bound), prefill (compute-bound), and mixed (chunked prefill with concurrent decode) workloads

**Performance on Llama 3 8B on TPU7x:**

| Metric | Achieved |
|--------|----------|
| Memory Bandwidth Utilization (decode) | **86% of hardware peak** |
| Model FLOPs Utilization (prefill) | **73% of hardware peak** |
| Token throughput vs pre-integration | **5× improvement** (since vLLM-TPU integration in February 2025) |

RPA is the **primary TPU attention backend in both vLLM and SGLang** (vLLM PR #13379).

The block table abstraction is preserved; only the kernel implementation changes for TPU hardware characteristics.

(source: raw/2026-04-24-ragged-paged-attention-tpu-arxiv.md)

## Relationship to Other Concepts
- Enables [Continuous Batching](continuous-batching.md) — without efficient memory, dynamic batching can't work
- Foundation for [Prefix Caching](../techniques/prefix-caching.md) — shared blocks make prefix reuse natural
- Managed by [KV Cache Management](kv-cache-management.md) subsystem

## Open Questions
- How does block size affect performance across different model architectures?
- What's the overhead of the block table lookup compared to contiguous memory access?
- What is the RPA decode MBU for larger models (70B+) where the KV cache is proportionally larger?
- Does RPA's distribution-aware compilation handle chunked-prefill+decode mixed batches as efficiently as pure-prefill or pure-decode?
- How does TPU7x RPA performance compare to equivalent GPU-side FlashAttention on H100 for the same model/context length?
