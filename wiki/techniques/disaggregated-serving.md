---
title: "Disaggregated Serving (Prefill-Decode Separation)"
tags: [architecture, serving, latency, scale, kv-connector, eplb, nixl]
created: 2026-04-14
updated: 2026-04-24
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-roadmap-q1-2026.md, raw/2026-04-20-streamserve-arxiv-2604-09562.md, raw/2026-04-20-prefill-as-a-service-arxiv-2604-15039.md, raw/2026-04-24-vllm-v020-release.md]
related: [concepts/chunked-prefill.md, concepts/continuous-batching.md, techniques/tensor-parallelism.md, techniques/speculative-decoding.md, concepts/kv-cache-management.md, concepts/deepseek-v4-attention.md]
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

## StreamServe: Adaptive Speculative Flows (arXiv 2604.09562, April 2026)

StreamServe combines disaggregated P/D with runtime-adaptive speculative decoding depth, addressing two separate inefficiencies in one architecture.

**Four components**:
- **StreamScheduler** — routes requests to stream pairs based on sequence-length prediction, memory headroom, and SLO state
- **FlowGuard** — multi-signal router tracking per-stream queue depth, KV pressure, and TTFT violations; dynamically redirects requests across pairs
- **PipeServe-Engine** — disaggregated P/D backend; each "stream pair" is one prefill GPU + one decode GPU; KV transferred asynchronously
- **SpecuStream** — tunes speculation depth K online from live acceptance rates; K increases when acceptance is high, decreases when acceptance drops (e.g., mid-reasoning-problem entropy spikes)

**Hardware**: 4× A800 40GB GPUs as 2 stream pairs. Benchmarks: ALPACA, GSM8K, HUMANEVAL, SUM (320 total queries).

| Benchmark | TP-vLLM | StreamServe | Delta |
|-----------|---------|-------------|-------|
| GSM8K throughput | 241 tok/s | 264 tok/s | +9.5% |
| GSM8K latency | 3.50 s | 0.30 s | −91% |
| SUM peak throughput | — | 2235 tok/s | — |
| Overall latency | baseline | 11–18× lower | — |

**Limitations**: Small evaluation (320 queries, A800 only); baseline is TP vLLM without native disaggregation or fixed-K spec decode, not vLLM's own P/D + spec decode stack; no H100/B200 numbers.

**Relation to vLLM**: vLLM's P/D stack (KV Connector + Dynamo/llm-d) implements the same architectural separation. SpecuStream's adaptive K is not in vLLM — vLLM uses a static K per job. See [Speculative Decoding](speculative-decoding.md) for detail on SpecuStream.

(source: raw/2026-04-20-streamserve-arxiv-2604-09562.md)

## Prefill-as-a-Service: Cross-Datacenter P/D (arXiv 2604.15039, April 2026)

Hybrid-attention architectures (mixing full-attention and linear-attention/SSM layers) produce far smaller KV caches than dense-attention models — only full-attention layers generate KV pairs. This unlocks cross-datacenter disaggregated serving over commodity Ethernet, which is impractical for dense-attention due to KV transfer volume.

**PrfaaS architecture** (from Moonshot AI / Kimi):
1. **Selective offloading policy** — only long-prompt requests sent to remote prefill cluster; short prompts handled locally
2. **Bandwidth-aware scheduler** — tracks real-time inter-datacenter BW; delays offloading if saturated
3. **Cache-aware placement** — repeated or prefix-matched prompts routed to prefill nodes that have cached their KV prefix
4. **Commodity Ethernet transport** — TCP/IP; no RDMA required

**Results** (internal 1T-parameter hybrid model):

| Baseline | Throughput vs PrfaaS |
|----------|---------------------|
| Homogeneous PD (same DC) | +54% |
| Naive heterogeneous (multi-DC, no PrfaaS) | +32% |

**Key implication**: As hybrid-attention models become standard (Mamba2, Jamba, Gemma2 sliding window, future architectures), the cost model for P/D disaggregation changes fundamentally. Prefill tiers can be globally shared across geographies. vLLM's KV Connector handles intra-DC transport; cross-DC requires a network-aware transport layer not yet in vLLM.

**Limitation**: Evaluated on one undisclosed model; commodity Ethernet cross-DC latency adds 10–100ms RTT, hurting interactive TTFT.

(source: raw/2026-04-20-prefill-as-a-service-arxiv-2604-15039.md)

## vLLM v0.20.0 KV Infrastructure Updates (April 23, 2026)

### 3FS KVConnector
A new KV Connector targeting **Bytedance 3FS** (distributed file system) was added. 3FS is a high-throughput storage system; the connector enables KV cache sharing and persistence across vLLM worker nodes using 3FS as the transport layer. This is the first non-network (file-system-backed) KV connector in vLLM. Useful for checkpoint-style KV reuse across restarts, or for disaggregated deployments where workers share a high-speed storage fabric rather than direct GPU-GPU links.

(source: raw/2026-04-24-vllm-v020-release.md)

### EPLB (Expert Parallel Load Balancing) Communication Update
EPLB (Expert Parallel Load Balancing), which dynamically redistributes MoE expert weights across GPUs during serving, gained an alternative communication strategy for weight exchange in v0.20.0. Additionally, router record tracking for prefill mapping enables smarter load-balancing decisions. **Asyncio infrastructure removed from Async EPLB (PR #40730, April 23-24)**: replaces asyncio scheduling overhead with direct synchronous or CUDA-async calls, reducing EPLB scheduling latency.

(source: raw/2026-04-24-vllm-v020-release.md, raw/2026-04-24-vllm-prs-apr23-24.md)

### Nixl 0.10.1
The Nixl KV transfer library (used for high-speed cross-node KV migration in disaggregated P/D) was bumped to version 0.10.1. Specific changes not described in available release notes.

(source: raw/2026-04-24-vllm-v020-release.md)

### DeepSeek V4 and Disaggregated Serving
DeepSeek V4-Pro (1.6T params, 1M context) is the first model where vLLM explicitly documented disaggregated serving as a required deployment mode, not an optimization. The KV Connector API handles the hybrid KV cache (CSA compressed blocks + HCA compressed blocks + sliding window), which has different transfer semantics from dense-attention KV. See [DeepSeek V4 Attention](../concepts/deepseek-v4-attention.md).

(source: raw/2026-04-24-deepseek-v4-vllm.md)

## Open Questions
- What's the break-even point where disaggregation outperforms co-located serving?
- How does vLLM-Omni's multi-replica staging compare to purpose-built PD systems?
- Does StreamServe's SpecuStream adaptive K generalize to P-EAGLE (where K is a training-time hyperparameter, not a runtime variable)?
- At what hybrid-attention KV size does cross-datacenter P/D (PrfaaS) become competitive with intra-datacenter serving in terms of TTFT?
- When does vLLM's KV Connector gain a network-aware transport layer for cross-DC scenarios?
- What is the throughput of the 3FS KVConnector relative to NVLink/InfiniBand-based connectors?
- Does EPLB's alternative communication strategy resolve the weight exchange bottleneck at high expert parallelism (EP=64+)?
- What is the KV transfer volume for DeepSeek V4-Pro (1M context) in disaggregated serving, given the hybrid compressed/uncompressed KV layout?
