---
title: "vLLM — Ramp-Up Recap (Q&A session)"
source_type: conversation
source_url: N/A (internal ramp-up recap PDF)
collected: 2026-04-14
tags: [vllm, overview, paged-attention, continuous-batching, chunked-prefill, benchmarks, glossary]
---

# vLLM — Ramp-Up Recap

## What Is vLLM?

vLLM is an open-source **inference engine** for serving large language models in production. "Inference" means using a trained model to produce outputs (as opposed to training). vLLM does not make the model smarter — it makes the **serving infrastructure** efficient so the model can handle many concurrent users with low latency.

Analogy: the model is the chef, the inference engine is the kitchen. vLLM makes the kitchen run smoothly so the chef can cook for more customers at once.

- Current version: **v0.19.0** (April 2026)
- Origin: UC Berkeley
- 75K+ GitHub stars
- Docs: https://docs.vllm.ai/
- GitHub: https://github.com/vllm-project/vllm
- Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"

## Why LLM Serving Is Hard

1. **Autoregressive generation is sequential.** Each output token depends on all previous tokens. Each step requires a forward pass through the entire model. You cannot parallelize generation of a single response.
2. **The KV cache dominates memory.** During generation, the model stores key-value tensors for every token it has seen. For large models with long contexts, this cache consumes tens of gigabytes of GPU memory.
3. **Requests are unpredictable.** Prompt lengths range from 50 to 32K+ tokens. Response lengths vary just as wildly. The server must handle this without wasting GPU cycles on idle memory or blocked requests.

## The LLM Pipeline: Prefill vs. Decode

**Prefill Phase** — Processing the input prompt.
All input tokens are known upfront, so they are processed in parallel in one large matrix operation. This phase is **compute-bound** (GPU arithmetic units are the bottleneck). **TTFT** (Time To First Token) measures how long this takes.

**Decode Phase** — Generating the response, one token at a time.
Each token requires reading the entire KV cache but only produces one new token. This phase is **memory-bandwidth-bound** (the GPU spends most of its time reading KV cache, not doing math). **ITL** (Inter-Token Latency) measures the time between consecutive tokens.

Target for fluid streaming: **TTFT < 200ms, ITL < 30ms**.

## Core Innovation #1: PagedAttention

**The problem:** Traditional engines pre-allocate a contiguous block of GPU memory for each request's KV cache, sized for the maximum possible sequence length (e.g., 8K tokens = 400MB). If the actual response is only 500 tokens (25MB), the remaining 375MB sits reserved but empty. This leads to **60–80% memory waste**.

**The solution:** Divide KV cache into small fixed-size blocks (default: 16 tokens each). Allocate blocks on demand as tokens are generated. Free blocks instantly when a request finishes.

**Key mechanisms:**
- **Block table** — per-request mapping from logical to physical blocks in GPU memory
- **On-demand allocation** — blocks allocated one at a time as tokens are produced (negligible latency)
- **Immediate deallocation** — finished requests return blocks to the free pool instantly
- **Reference counting** — enables memory sharing for parallel sampling (copy-on-write)

**Result:** Under 4% memory waste (vs. 60–80% before). **2–4x higher throughput** because far more concurrent requests fit in the same GPU memory.

**Key insight — statistical multiplexing:** Not every request hits the maximum at the same time, so a shared pool serves more requests than individual reservations. The total pool can actually be smaller than the sum of all worst-case reservations.

**Tradeoff:** If the pool runs out, vLLM must **preempt** — evict blocks to CPU or recompute. The `gpu_memory_utilization` parameter (default 0.9) controls how aggressively you bet on statistical sharing.

## Core Innovation #2: Continuous Batching

**The problem:** Old engines use **static batching** — collect a full batch, process all requests, wait for every request to finish, then start a new batch. Short responses sit idle while long ones complete.

**The solution:** The scheduler checks for new requests at every decode iteration. Finished requests are immediately evicted (blocks freed), waiting requests are immediately inserted.

**Important:** "Continuous batching" is a bit of a misnomer — it is really "no batching" or "streaming admission." There is no "batch 1" and "batch 2" anymore. There is just one evolving set of active requests that changes at every iteration:

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

The GPU still processes all active requests together in each forward pass (that is the "batch" part), but membership in that group is **fluid, not fixed**.

**Why this was not possible before:** Pre-allocated contiguous memory could not be partially freed. PagedAttention's block-level allocation/deallocation is what makes continuous batching work — the two innovations are **tightly coupled**.

## Core Innovation #3: Chunked Prefill

**The problem:** A request with a 32K-token prompt monopolizes the GPU during prefill. Every other request's decode freezes — **head-of-line blocking**.

**The solution:** Split the long prompt into smaller chunks, interleaved with decode steps for other requests.

**Tradeoff:** The long-prompt user's TTFT gets slightly worse (same total compute, spread across more iterations). Everyone else's latency dramatically improves. A fairness tradeoff — slightly penalize one heavy request to prevent it from starving dozens of lighter ones.

## GPU Memory Layout

- **Model weights** — parameters (e.g., 70B in FP16 ≈ 140GB). Fixed, loaded once.
- **KV cache** — attention state for all active requests. Managed by PagedAttention. Controlled by `gpu_memory_utilization` (fraction of remaining memory after weights).
- **Activations** — temporary tensors during forward passes. Relatively small, reused across iterations.

Every concurrent request gets its own KV cache allocation (via PagedAttention blocks). All requests share the same model weights and activation buffer.

## Key Parameters Cheat Sheet

| Parameter | Controls | Default | Notes |
|---|---|---|---|
| `max_num_seqs` | Max concurrent sequences | varies | Higher = more throughput, needs more KV cache |
| `max_num_batched_tokens` | Total tokens per forward pass | varies | Controls compute/memory tradeoff, chunk size |
| `gpu_memory_utilization` | Fraction of GPU for KV cache | 0.9 | Lower on shared GPUs; higher = more concurrency but more preemption risk |

## Where vLLM Sits Today (April 2026)

- **V1 engine** — default since v0.8.0. 1.7x throughput over original. Prefix caching nearly free (<1% overhead).
- **Model Runner V2** — ground-up rewrite of execution layer (March 2026). Default in v0.19.0; MRV1 remains as fallback.
- **Blackwell support** — full NVIDIA SM120 support since v0.15.1 (Feb 2026).

## Competitive Benchmarks

Source: Clarifai, GPT-OSS-120B on 2×H100.

| Engine | Strength | Weakness |
|---|---|---|
| vLLM | Highest throughput at high concurrency (4,741 tok/s), fastest TTFT | — |
| SGLang | Most stable ITL (4–21ms), better for multi-turn (RadixAttention) | Lower peak throughput |
| TensorRT-LLM | Best single-request throughput | Scales worse, requires compilation, slowest TTFT |

## Glossary

- **TTFT** — Time To First Token. Latency from prompt submission to first response token. Measures prefill speed.
- **ITL** — Inter-Token Latency. Time between consecutive output tokens during streaming. Measures decode speed.
- **KV cache** — Stored key-value tensors from attention layers for all tokens seen so far.
- **Prefill** — Processing the input prompt (parallel, compute-bound).
- **Decode** — Generating output tokens one at a time (sequential, memory-bandwidth-bound).
- **Preemption** — Evicting a request's KV blocks when memory runs out (swap to CPU or recompute).
- **TP** — Tensor Parallelism. Splitting model layers across multiple GPUs.
