---
title: "Continuous Batching"
tags: [scheduling, throughput, vllm-core]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-docs.md, raw/vllm-benchmarks-2026.md]
related: [concepts/paged-attention.md, concepts/chunked-prefill.md, techniques/disaggregated-serving.md]
---

# Continuous Batching

## Summary
Continuous batching (also called iteration-level batching) dynamically adds and removes requests from a batch at every decode step, rather than waiting for an entire batch to complete before starting the next one. This dramatically reduces latency and improves GPU utilization.

## How It Works
In **static batching** (the naive approach), the server waits until it has a full batch of requests, processes them all together, and waits for every sequence in the batch to finish before accepting new requests. Short sequences wait idle while long sequences complete.

In **continuous batching**, the scheduler checks for new requests at every iteration:
1. A request finishes (hits EOS or max length) → immediately evicted, its memory freed
2. A new request arrives → immediately inserted into the batch, prefill computed
3. The batch is always maximally utilized

This works hand-in-hand with [PagedAttention](paged-attention.md) — because memory is allocated per-block rather than pre-reserved, finished sequences instantly release their memory for new ones.

## Key Parameters
- `max_num_seqs` — caps concurrent sequences in a batch
- `max_num_batched_tokens` — caps total tokens per forward pass (controls memory/compute tradeoff)

## Relationship to Other Concepts
- Requires [PagedAttention](paged-attention.md) for efficient memory recycling
- [Chunked Prefill](chunked-prefill.md) prevents long prefills from blocking decode requests
- [Disaggregated Serving](../techniques/disaggregated-serving.md) takes this further by separating prefill and decode entirely

## Open Questions
- How does preemption policy affect tail latency under continuous batching? (vLLM Q2 2026 roadmap flags "avoid excessive preemption" as a known issue)
