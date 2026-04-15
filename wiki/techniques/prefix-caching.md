---
title: "Prefix Caching"
tags: [memory, kv-cache, throughput, caching]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-benchmarks-2026.md, raw/vllm-releases.md]
related: [concepts/paged-attention.md, concepts/kv-cache-management.md]
---

# Prefix Caching

## Summary
Prefix caching reuses computed KV cache blocks across requests that share the same prompt prefix (e.g., system prompts, few-shot examples). Instead of recomputing attention for the shared prefix every time, the cached blocks are simply referenced. In vLLM V1, this is nearly free — less than 1% overhead even at 0% hit rate.

## Problem It Solves
Many LLM serving workloads have shared prefixes: the same system prompt, the same few-shot examples, or multi-turn conversations where earlier messages are identical. Without caching, each request recomputes the full prefill, wasting both compute and memory.

## How It Works
1. KV cache blocks are hashed by their content (the tokens they represent)
2. When a new request arrives, the engine checks if any prefix blocks already exist in cache
3. Matching blocks are shared (via PagedAttention's reference counting), skipping prefill for those tokens
4. Only the novel suffix tokens need new computation

vLLM V1 made this essentially free by default — the overhead of cache checking is negligible, so it's always enabled.

## Benchmarks
| Metric | Without Prefix Cache | With Prefix Cache | Notes |
|--------|---------------------|-------------------|-------|
| TTFT (shared prefix) | baseline | up to 90%+ reduction | depends on prefix length |
| Overhead at 0% hit | — | <1% | V1 engine |
| Memory | 1x | shared blocks reused | saves proportional to prefix length |

## Trade-offs
- Essentially no downside in V1 (overhead is negligible)
- Effectiveness depends on workload — random prompts with no shared prefix get no benefit
- SGLang's RadixAttention may be better for complex multi-turn prefix patterns

## When to Use
- Always (it's on by default in V1 and nearly free)
- Especially valuable for: multi-turn chat, system prompts, few-shot prompting, batch processing with shared context

## Open Questions
- How does vLLM's prefix caching compare to SGLang's RadixAttention in multi-turn workloads?
- What's the interaction between prefix caching and FP8 KV cache quantization?
