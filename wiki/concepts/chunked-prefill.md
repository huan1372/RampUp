---
title: "Chunked Prefill"
tags: [scheduling, latency, prefill, vllm-core]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-docs.md]
related: [concepts/continuous-batching.md, techniques/disaggregated-serving.md, concepts/kv-cache-management.md]
---

# Chunked Prefill

## Summary
Chunked prefill splits long-context prompt processing into smaller chunks so that decode (token generation) requests aren't starved while a massive prefill monopolizes the GPU. This prevents the "head-of-line blocking" problem where one long-prompt request freezes latency for all concurrent shorter requests.

## How It Works
Without chunking, a single request with a 32K-token prompt would occupy the GPU for the entire prefill computation before any decode tokens can be generated for other requests in the batch.

With chunked prefill:
1. The long prompt is split into chunks (controlled by `max_num_batched_tokens`)
2. Each chunk is processed in one forward pass, interleaved with decode steps for other requests
3. The KV cache for the prompt is built incrementally across multiple iterations

The tradeoff: chunked prefill adds some overhead to total prefill time for that one request, but dramatically improves TTFT and ITL for all other concurrent requests.

## Key Parameters
- `max_num_batched_tokens` — effectively controls chunk size
- `enable_chunked_prefill` — toggle (default: enabled in V1 engine)

## Relationship to Other Concepts
- Complements [Continuous Batching](continuous-batching.md) — without chunking, long prefills block the batch
- Alternative to [Disaggregated Serving](../techniques/disaggregated-serving.md), which solves the same problem by running prefill on separate hardware
- AMD ROCm supports a split prefill-decode attention mode (`VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`)

## Open Questions
- What's the optimal chunk size for different model sizes and context lengths?
- How does chunked prefill interact with speculative decoding?
