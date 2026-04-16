---
title: "Continuous Batching"
tags: [scheduling, throughput, vllm-core]
created: 2026-04-14
updated: 2026-04-15
sources: [raw/vllm-docs.md, raw/vllm-benchmarks-2026.md, raw/2026-04-14-vllm-rampup-recap.md]
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

This works hand-in-hand with [PagedAttention](paged-attention.md) — because memory is allocated per-block rather than pre-reserved, finished sequences instantly release their memory for new ones. **The two innovations are tightly coupled**: pre-allocated contiguous memory could not be partially freed, which is why continuous batching was not feasible before PagedAttention (source: raw/2026-04-14-vllm-rampup-recap.md).

## Static vs. Continuous Batching (Timeline)

```
Static Batching:
Batch 1: [req1, req2, req3] → all run until ALL finish
Batch 2: [req4, req5, req6] → cannot start until batch 1 done

Continuous Batching:
Step 1: active = [req1, req2, req3]
Step 2: active = [req1, req2, req3]  ← req1 finishes
Step 3: active = [req2, req3, req4]  ← req4 joins
Step 4: active = [req2, req3, req4]  ← req3 finishes
Step 5: active = [req2, req4, req5]  ← req5 joins
```

The name "continuous batching" is a mild misnomer — it is really **streaming admission**.
There is no fixed "batch 1" and "batch 2" anymore; there is one evolving set of active
requests whose membership changes every iteration. The GPU still processes all active
requests together in each forward pass (that is the "batch" part), but membership is
**fluid, not fixed** (source: raw/2026-04-14-vllm-rampup-recap.md).

## Key Parameters
- `max_num_seqs` — caps concurrent sequences in a batch
- `max_num_batched_tokens` — caps total tokens per forward pass (controls memory/compute tradeoff)

## Relationship to Other Concepts
- Requires [PagedAttention](paged-attention.md) for efficient memory recycling
- [Chunked Prefill](chunked-prefill.md) prevents long prefills from blocking decode requests
- [Disaggregated Serving](../techniques/disaggregated-serving.md) takes this further by separating prefill and decode entirely

## Open Questions
- How does preemption policy affect tail latency under continuous batching? (vLLM Q2 2026 roadmap flags "avoid excessive preemption" as a known issue)
