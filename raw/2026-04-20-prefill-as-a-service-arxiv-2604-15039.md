---
title: "Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.15039
collected: 2026-04-20
tags: [disaggregated-serving, kv-cache, prefill, cross-datacenter, hybrid-attention]
---

# Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

**arXiv**: 2604.15039  
**Submitted**: April 2026  
**Affiliation**: Moonshot AI (Kimi)

## Core Claim

Hybrid-attention architectures (combining full-attention layers with linear-attention or SSM layers) produce dramatically smaller KV caches than dense-attention models. This small KV cache unlocks **cross-datacenter disaggregated prefill-decode** over commodity Ethernet — something impractical with dense-attention models because their KV cache transfer volume saturates inter-datacenter bandwidth.

## Background

Standard PD disaggregation (DistServe, vLLM P/D) assumes prefill and decode are co-located within a high-bandwidth network domain (NVLink or high-speed IB). The constraint is that KV cache transfer (prefill → decode) must be fast enough to not become the bottleneck. For dense-attention LLMs, KV cache is O(layers × seq_len × heads × head_dim) — enormous for long contexts. This locks prefill and decode into the same datacenter.

Hybrid-attention models replace some or all attention layers with linear-attention / SSM layers. These layers use fixed-size recurrent state rather than KV cache, drastically reducing KV cache volume (only full-attention layers generate KV pairs). For a model with few full-attention layers, total KV size may be 5–20× smaller than a fully-dense equivalent.

## Prefill-as-a-Service (PrfaaS) Architecture

Selective offloading: PrfaaS routes only prefill-heavy requests (long prompts) to remote prefill-specialized clusters. Short-prompt requests are handled locally.

**Four system-side components**:
1. **Selective offloading policy**: decides which requests to offload based on prompt length, current remote prefill cluster load, and estimated KV transfer cost vs local prefill cost
2. **Bandwidth-aware scheduler**: tracks real-time inter-datacenter bandwidth utilization; delays or cancels offloading if bandwidth is saturated
3. **Cache-aware request placement**: routes repeated or prefix-matched prompts to the prefill cluster that has cached their KV prefix, amortizing transfer across repeated queries
4. **Commodity Ethernet transport**: uses standard TCP/IP rather than RDMA; practical for cross-datacenter deployment without specialized hardware

## Results

Case study: internal 1T-parameter hybrid model (architecture details not disclosed).

| Baseline | Throughput vs PrfaaS |
|----------|---------------------|
| Homogeneous PD (all same GPU) | +54% higher with PrfaaS |
| Naive heterogeneous (ad-hoc multi-DC) | +32% higher with PrfaaS |

The throughput gain comes from: (1) prefill clusters can be optimized independently (compute-dense GPUs for prefill, memory-bandwidth-dense GPUs for decode), and (2) cache-aware placement enables high KV prefix cache hit rates across the prefill tier.

Cross-datacenter bandwidth consumption: described as "modest" — exact GB/s figures not in abstract.

## Key Implications

- **Hybrid-attention unlocks elastic prefill scaling**: as models shift toward hybrid architectures (Mamba2, Jamba, Gemma2 with sliding window, future architectures), the PD disaggregation cost model changes fundamentally
- **Prefill cluster can be globally shared**: multiple decode clusters in different datacenters could share a single large prefill cluster, amortizing compute across geographies
- **vLLM relevance**: vLLM's KV Connector API already supports P/D separation within a datacenter; PrfaaS's cross-datacenter extension would require a network-aware transport layer on top

## Limitations

- Evaluated only on one internal hybrid model; results may not generalize to different hybrid architectures with different full-attention layer ratios
- Commodity Ethernet latency could hurt TTFT for interactive use cases (cross-datacenter RTT adds 10–100ms vs intra-datacenter)
- Bandwidth-aware scheduler complexity adds operational overhead

## Relation to Prior Work

- **DistServe (arXiv 2401.09670)**: original PD disaggregation paper; assumes same datacenter
- **vLLM P/D via NIXL**: production implementation within same network domain
- **LMCache**: KV cache layer for sharing across requests; complementary (PrfaaS adds cross-datacenter dimension)
- **TraCT (arXiv 2512.18194)**: rack-scale KV sharing via CXL; different approach (CXL hardware vs commodity Ethernet)
