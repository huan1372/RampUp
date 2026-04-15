---
title: "Disaggregated Serving (Prefill-Decode Separation)"
tags: [architecture, serving, latency, scale]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-roadmap-q1-2026.md]
related: [concepts/chunked-prefill.md, concepts/continuous-batching.md, techniques/tensor-parallelism.md]
---

# Disaggregated Serving

## Summary
Disaggregated serving separates the prefill (prompt processing) and decode (token generation) phases onto different hardware. Prefill is compute-bound and benefits from high-bandwidth GPUs; decode is memory-bound and benefits from high-capacity KV cache. By specializing hardware, both phases run more efficiently.

## Problem It Solves
Prefill and decode have fundamentally different resource profiles. Mixing them on the same GPU means neither is optimally served — long prefills block decodes (hurting latency), and decode's low arithmetic intensity underutilizes the GPU during generation.

## How It Works
1. **Prefill nodes** process incoming prompts and generate the initial KV cache
2. KV cache is transferred to **decode nodes** (via NVLink, InfiniBand, or network)
3. **Decode nodes** handle autoregressive token generation
4. Each node type can be independently scaled

This is sometimes called the "PD" (Prefill-Decode) architecture. vLLM's Q2 2026 roadmap extends this to "vLLM-Omni" where individual stages can be initialized with different numbers of replicas.

## Implementation in vLLM
- Supported via the KV Connector API
- `llm-d` and `Dynamo` ecosystem integration for routing
- vLLM-router can sit between API server and engine core for traffic management
- Q2 2026: focus on GB200 with NVLink, CPU unified memory, and multi-stream concurrency

## Trade-offs
- Adds network transfer overhead for KV cache migration
- More complex infrastructure (two node types, routing logic)
- Only beneficial at scale — single-GPU or small deployments don't benefit

## When to Use
- Large-scale production deployments with heterogeneous hardware
- Workloads with highly variable prompt lengths (some very long prefills)
- When TTFT SLOs are strict and long prefills are unacceptable

## Open Questions
- What's the break-even point where disaggregation outperforms co-located serving?
- How does vLLM-Omni's multi-replica staging compare to purpose-built PD systems?
