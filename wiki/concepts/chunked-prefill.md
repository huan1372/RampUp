---
title: "Chunked Prefill"
tags: [scheduling, latency, prefill, vllm-core]
created: 2026-04-14
updated: 2026-04-28
sources: [raw/vllm-docs.md, raw/2026-04-14-vllm-rampup-recap.md, raw/2026-04-28-vllm-prs-apr27-28.md]
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

**Framing it as fairness:** Chunked prefill is explicitly a fairness mechanism — slightly penalize one heavy request to prevent it from starving dozens of lighter ones (source: raw/2026-04-14-vllm-rampup-recap.md). The phenomenon it prevents is called **head-of-line blocking**: in decode, every request requires reading the full KV cache but only produces one token, so a monopolizing prefill freezes everyone else's ITL.

## Why Prefill Needs Special Handling

Prefill and decode have opposite resource profiles:
- **Prefill** is compute-bound (GPU arithmetic is the bottleneck, all input tokens processed in parallel)
- **Decode** is memory-bandwidth-bound (most time spent reading KV cache, only one new token produced per step)

Targets for fluid streaming: **TTFT < 200ms, ITL < 30ms**. Without chunking, a 32K-token prefill can blow past both targets for every other request in the batch (source: raw/2026-04-14-vllm-rampup-recap.md).

## Key Parameters
- `max_num_batched_tokens` — effectively controls chunk size
- `enable_chunked_prefill` — toggle (default: enabled in V1 engine)

## Relationship to Other Concepts
- Complements [Continuous Batching](continuous-batching.md) — without chunking, long prefills block the batch
- Alternative to [Disaggregated Serving](../techniques/disaggregated-serving.md), which solves the same problem by running prefill on separate hardware
- AMD ROCm supports a split prefill-decode attention mode (`VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`)

## StepPool Embedding Alignment Fix (PR #41049, April 28, 2026)

For embedding models processing long sequences (4K+ tokens), chunked prefill in `StepPool.forward` had a correctness bug: `pooled_data.append(data)` was placed outside the `if step_tag_id` conditional block, causing None to be appended twice per unfinished chunk. This misaligned pooling output with batch requests.

**Fix:** Move the append inside the conditional block. Verified on TPU with 4K+ token sequences. No performance change — correctness fix only.

**Scope:** Affects embedding models (e.g., retrieval, classification) using chunked prefill. Standard generative inference (`step_tag_id` not set) is unaffected.

(source: raw/2026-04-28-vllm-prs-apr27-28.md)

## Open Questions
- What's the optimal chunk size for different model sizes and context lengths?
- How does chunked prefill interact with speculative decoding?
