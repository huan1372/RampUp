---
title: "Speculative Decoding"
tags: [latency, throughput, decoding, speculation, monte-carlo, smc, distributed-inference, edge]
created: 2026-04-14
updated: 2026-04-28
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md, raw/2026-04-15-p-eagle-blog.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-19-calibrated-speculative-decoding-arxiv.md, raw/2026-04-20-specguard-arxiv-2604-15244.md, raw/2026-04-20-streamserve-arxiv-2604-09562.md, raw/2026-04-21-vllm-v0191-release.md, raw/2026-04-24-vllm-v020-release.md, raw/2026-04-24-vllm-prs-apr23-24.md, raw/2026-04-24-smc-sd-arxiv.md, raw/2026-04-26-dip-sd-arxiv-2604-20919.md, raw/2026-04-28-vllm-prs-apr27-28.md]
related: [concepts/model-runner-v2.md, concepts/continuous-batching.md, techniques/disaggregated-serving.md]
---

# Speculative Decoding

## Summary
Speculative decoding uses a small, fast "draft" model to predict multiple tokens ahead, then verifies them in a single forward pass of the large "target" model. Correct predictions are accepted for free; incorrect ones are rejected and regenerated. This can significantly reduce latency without changing output quality.

## Problem It Solves
Autoregressive decoding is inherently sequential — each token depends on the previous one. For large models, each forward pass has high latency even though the GPU is underutilized (especially for single-sequence decode). Speculative decoding amortizes the cost by verifying multiple tokens per forward pass.

## How It Works
1. **Draft phase**: a small model generates K candidate tokens quickly
2. **Verify phase**: the target model processes all K tokens in one forward pass
3. **Accept/reject**: using a rejection sampler, tokens are accepted if they match what the target model would have produced. The first rejected token is resampled from the target.
4. On average, multiple tokens are accepted per iteration, yielding a speedup proportional to the acceptance rate.

## Implementation in vLLM
- **EAGLE-3** supported (Q1 2026 roadmap: "Support and test EAGLE-3 thoroughly")
- **Zero-bubble async scheduling** — speculative decoding now works with async scheduling for overlap, default in v0.19.0 (PR #32951)
- **MRV2 integration** — rejection sampler with greedy/logprobs support (PRs #37238, #37237)
- **Multi-modal spec decode** — embeddings for vision models (PR #36097)
- **NGram GPU spec decode** (v0.18.0) — NGram drafting now runs on GPU, compatible with async scheduler

### New in v0.20.0 (April 23, 2026)

**CPU draft-model speculative decoding**: v0.20.0 enables CPU-hosted draft models in vLLM's speculative pipeline for the first time. Previously, all draft models had to reside on GPU. CPU draft models are slower but allow decoupling the draft model from GPU memory, useful when GPU memory is tight. Quality of the speculative pipeline depends on draft-target alignment; CPU draft throughput limits maximum useful K. (source: raw/2026-04-24-vllm-v020-release.md)

**Full CUDA graph for FlexAttention and Eagle prefill**: Full CUDA graph capture is now supported for both FlexAttention and Eagle (EAGLE-3 drafter) prefill paths. Previously, Eagle prefill ran without CUDA graphs, causing kernel launch overhead on every step. With full graph capture, the Eagle prefill is amortized into the graph replay, reducing latency for each speculative step. (source: raw/2026-04-24-vllm-v020-release.md)

**PR #40654 — Eliminate seq_lens_cpu GPU→CPU Sync** (April 23-24, 2026): Removes an unnecessary GPU-to-CPU synchronization in sequence length computation within the speculative decode pipeline. GPU-CPU syncs create pipeline bubbles — the GPU must drain before the CPU reads. Removal reduces TTFT and TPOT overhead in speculative decode paths. (source: raw/2026-04-24-vllm-prs-apr23-24.md)

**PR #40662 — Unified Acceptance Rate Metric V1/V2** (April 23-24, 2026): Acceptance rate was computed differently in V1 and MRV2 backends. Unification enables consistent cross-backend benchmarking. (source: raw/2026-04-24-vllm-prs-apr23-24.md)

## P-EAGLE: Parallel Speculative Decoding
P-EAGLE (shipped in vLLM v0.16.0+) transforms EAGLE's autoregressive draft generation into a single parallel forward pass. (source: raw/2026-04-15-p-eagle-blog.md)

### How It Differs from EAGLE
Standard EAGLE generates K draft tokens sequentially (K forward passes of the draft model). P-EAGLE generates all K tokens in one pass using learnable "mask" parameters as placeholders for unknown future positions — those parameters are learned during drafter training.

### Performance vs. EAGLE-3

| Benchmark   | Concurrency | P-EAGLE speedup over EAGLE-3 |
|-------------|-------------|------------------------------|
| SPEED-Bench | 1           | **1.69×**                    |
| MT-Bench    | 1           | 1.55×                        |
| HumanEval   | 1           | 1.55×                        |
| MT-Bench    | 64          | 1.05× (diminishes at scale)  |
| HumanEval   | 64          | 1.23×                        |

**Acceptance length at K=7**: P-EAGLE shows ~30-31% higher acceptance rates than EAGLE-3 (HumanEval: 3.94 vs 3.03; SPEED-Bench: 3.38 vs 2.59)

**Optimal speculation depth**: P-EAGLE peaks at K=7; EAGLE-3 maxes at K=3. Larger K is economical because drafting costs a flat single forward pass regardless of K.

Pre-trained drafter models available for: GPT-OSS 120B, GPT-OSS 20B, Qwen3-Coder 30B.

## Eagle3 + Gemma4 (vLLM v0.19.1)

vLLM v0.19.1 (April 18, 2026) added Eagle3 speculative decoding support for Gemma4 models via PR #39450. This is the first Eagle-family drafter for a Google-origin MoE+multimodal architecture. Key context:
- Gemma4 is a hybrid dense+MoE model with multimodal and reasoning capabilities
- The Eagle3 drafter runs alongside the Gemma4 target model, requiring the additional drafter to fit in GPU memory
- Requires `transformers>=5.5.3` (resolved by the v0.19.1 dependency upgrade)
- Quantized MoE inference for Gemma4 (PR #39045) is available in the same release, allowing the Gemma4 target model to run in FP8/quantized mode while Eagle3 provides speculative drafts

(source: raw/2026-04-21-vllm-v0191-release.md)

## Trade-offs
- Requires maintaining a second (draft) model in memory
- Acceptance rate varies by task — highly predictable text benefits most
- Adds complexity to the serving pipeline

## When to Use
- Latency-sensitive applications (interactive chat, coding assistants)
- When the draft model has a high acceptance rate for the target workload
- When GPU is underutilized during decode (low-concurrency scenarios)

## Calibrated Speculative Decoding (CSD) — Research (arXiv 2604.13634, April 2026)

CSD is a training-free framework that addresses **false rejections** — cases where the draft model's token is semantically correct but lexically diverges from the target model's top token, causing the standard rejection sampler to discard a valid output.

**Two modules**:
1. **Online Correction Memory (OCM)**: accumulates rejected (draft token, context) pairs at runtime; identifies recurring divergence patterns and proposes "rescue candidates" drawn from historical rejections
2. **Semantic Consistency Gating (SCG)**: validates rescue candidates via a probability ratio test (draft / target), admitting only tokens within a calibrated threshold — prevents hallucinated rescues

**Results**: Peak throughput speedup of **2.33×** over baseline speculative decoding across diverse LLMs (exact models not specified). Training-free — plugs into any existing draft-verify pipeline.

**vLLM integration status**: Not yet integrated. Authors explicitly note vLLM integration as needed for production deployment. Implementation requires changes to vLLM's rejection sampler (`vllm/model_executor/layers/spec_decode/`) and addition of a per-worker or global correction memory store.

(source: raw/2026-04-19-calibrated-speculative-decoding-arxiv.md)

## SpecGuard: Step-Level Verification for Reasoning (arXiv 2604.15244, April 2026)

Standard spec decode is token-centric: a rejected token triggers rollback, but an accepted erroneous reasoning step propagates forward corrupting the chain. SpecGuard adds step-level verification using model-internal signals only.

**Mechanism**:
1. Draft model samples multiple candidate steps (not just one)
2. Self-consistency scoring selects the most consistent candidate
3. An ensemble of lightweight verifiers (distilled from target model hidden states) validates semantic soundness
4. Accept or recompute the step from target; only the step is recomputed, not the full sequence

**Results** (multiple reasoning benchmarks, vs standard SD and reward-guided SD):

| Metric | vs Standard SD | vs Reward-guided SD |
|--------|---------------|---------------------|
| Accuracy | +3.6% | better |
| Latency | −11% | better |

No external reward model — overhead is proportional to draft model, not target.

**vLLM integration**: Not yet implemented. Requires step-boundary detection + multi-candidate draft sampling in vLLM's spec decode framework.

**Comparison with other approaches**: SpecGuard targets reasoning correctness; [CSD](speculative-decoding.md#calibrated-speculative-decoding-csd--research-arxiv-260413634-april-2026) targets false-rejection throughput; [P-EAGLE](speculative-decoding.md#p-eagle-parallel-speculative-decoding) targets draft-generation speed. Complementary, not competing.

(source: raw/2026-04-20-specguard-arxiv-2604-15244.md)

## SpecuStream: Runtime-Adaptive Speculation Depth (StreamServe, arXiv 2604.09562)

A component of the [StreamServe](disaggregated-serving.md#streamserve-adaptive-speculative-flows-arxiv-260409562-april-2026) disaggregated serving system. SpecuStream adjusts speculation depth K online from live acceptance rate signals.

**Problem**: Static K is suboptimal for reasoning tasks — during early problem setup, acceptance rates are high (predictable boilerplate); during mid-problem derivation steps, entropy spikes and acceptance drops, wasting draft computation at the original K.

**Mechanism**: Per-stream running average of accepted tokens per verification step. K increases when rate exceeds threshold; K decreases when rate falls below floor. Adjustments happen per request batch without stopping inference.

**Effect**: Combined with disaggregated serving (PipeServe-Engine), StreamServe achieves 9.5% throughput improvement on GSM8K (264 tok/s vs 241 tok/s TP vLLM) and 11–18× latency reduction. See [Disaggregated Serving](disaggregated-serving.md) for full benchmark table.

(source: raw/2026-04-20-streamserve-arxiv-2604-09562.md)

## Sequential Monte Carlo Speculative Decoding (SMC-SD) — Research (arXiv 2604.15672, April 2026)

SMC-SD replaces speculative decoding's **rejection sampling** with **importance-weighted particle resampling** (Sequential Monte Carlo). The key problem with rejection sampling: when draft and target diverge, the draft block is truncated at the first rejection, wasting all subsequent draft tokens. SMC-SD avoids this by treating each decode step as a population of N draft particles, assigning importance weights from the likelihood ratio (target/draft), and resampling — no rollback, no wasted draft compute.

**Key properties:**
- **No rollback**: verification is a fixed-size vectorized operation regardless of draft quality
- **Approximate but bounded**: principled per-step approximation error bounds; output distribution differs slightly from the exact target distribution
- **Uses idle compute**: LLM inference is memory bandwidth-bound; particle scoring comes nearly free during the memory-bound attention ops

**Performance:**

| Metric | vs Standard Speculative Decoding | vs Autoregressive Decoding |
|--------|----------------------------------|---------------------------|
| Speedup | **2.36×** | **5.2×** |
| Accuracy | Within 3% of target | Within 3% of target |

Benchmarks: reasoning, instruction-following, and coding tasks. Authors: Yahya Emara et al. (Cornell/MIT/ETH Zürich).

**Comparison with other training-free spec decode improvements:**
- **CSD (arXiv 2604.13634):** addresses false rejections via online correction memory — different mechanism, similar goal
- **P-EAGLE:** improves draft generation speed; SMC-SD improves the verification/acceptance step — complementary
- **SpecGuard:** targets reasoning step correctness; orthogonal

No vLLM integration. Implementation requires N-particle draft generation and importance weighting in `vllm/model_executor/layers/spec_decode/`.

(source: raw/2026-04-24-smc-sd-arxiv.md)

## DiP-SD: Distributed Pipelined Speculative Decoding at the Edge (arXiv 2604.20919, April 2026)

DiP-SD moves speculative decoding to a **distributed edge setting**: on-device small LLMs (phones/tablets) serve as draft models; a centralized edge server runs the large target model for batch verification. This is architecturally distinct from all other SD variants in this KB, which assume co-located draft and target.

**Two parallelism dimensions:**
1. **Device-level distributed drafting**: N users draft K tokens in parallel on their devices simultaneously — server's verification cost amortizes across the N-user batch
2. **Phase-level draft-verify pipelining**: device drafting for batch i+1 overlaps with server verification of batch i, eliminating the idle "wait for verify" phase in naive distributed SD

**Joint optimization**: DiP-SD co-optimizes user-to-batch assignment, per-user speculation length K, and pipeline batch count under per-user latency constraints. Objective: maximize expected accepted tokens per unit time.

**Claim**: "first SD scheme to exploit batch-level pipeline parallelism between distributed drafting and centralized verification" (source: raw/2026-04-26-dip-sd-arxiv-2604-20919.md)

**Authors:** Yaodan Xu, Sheng Zhou, Zhisheng Niu. Submitted April 22, 2026.

**vLLM relevance:** Not integrated into vLLM; the edge deployment model (device draft + server verify) differs from vLLM's co-located assumption. However, the per-request adaptive K insight aligns with open questions about per-request K tuning in vLLM's static-K spec decode pipeline.

**Limitation in sourcing:** arXiv HTML returned 403; specific speedup numbers unavailable. Paper reports "superior performance across all test points" vs. naive distributed SD and greedy batching.

(source: raw/2026-04-26-dip-sd-arxiv-2604-20919.md)

## Eagle Prefill Metadata Skip (PR #40410, April 27, 2026)

During Eagle speculative decoding in MRV2, attention metadata was rebuilt three times per step (target, Eagle prefill, draft decode). PR #40410 eliminates the redundant rebuild in the Eagle prefill phase by passing the target model's pre-built `CapturedAttentionState` to `PrefillEagleCudaGraphManager`. Draft decode continues to build independently.

**Result:** ~5–10% end-to-end latency improvement for Eagle speculative decode paths.

This is distinct from the v0.20.0 change that enabled CUDA graph capture for Eagle prefill (reducing kernel launch overhead): that change captured the graph; this change removes redundant work within each graph capture cycle.

(source: raw/2026-04-28-vllm-prs-apr27-28.md)

## Independent Drafter Attention Backend Selection (PR #39930, April 28, 2026)

Previously, the drafter model's attention backend was forced to match the target model's backend — causing failures when the two models had incompatible attention requirements (MLA drafter with GQA target; DFlash requiring non-causal attention not supported by the target's backend).

**Change:** New `--speculative-config.attention_backend` option. Unlike `--moe-backend`, the drafter does **not** inherit the target's backend when unspecified, because incompatibilities are common. Both backends auto-select independently.

**Behavior:**
- Neither specified → both auto-select independently
- Target specified, drafter not → drafter still auto-selects (no inheritance)
- Drafter specified, target not → each uses its own
- Both specified → each uses its own

**Concrete fix:** Resolves `ValueError: Selected backend TRITON_ATTN is not valid ... Reason: non-causal attention not supported` when using DFlash backends as drafter.

(source: raw/2026-04-28-vllm-prs-apr27-28.md)

## Open Questions
- What is the throughput of CPU draft models vs GPU draft models, and at what draft model size does CPU become impractical?
- Does CUDA graph Eagle prefill improve latency for all Eagle variants (Eagle-1, Eagle-2, Eagle-3, P-EAGLE)?
- What's the best draft model selection strategy for a given target model?
- How does EAGLE-3 compare to vanilla speculative decoding in practice?
- How does spec decode interact with chunked prefill scheduling?
- Why does P-EAGLE's advantage diminish at high concurrency (c=64)? Is this a batching overhead issue or acceptance rate regression?
- When will P-EAGLE drafter models be available for more target models beyond GPT-OSS and Qwen3-Coder?
- Does CSD's 2.33× peak speedup hold at high concurrency, or does it share P-EAGLE's diminishing returns pattern?
- What is the memory overhead of OCM's correction history at serving scale (thousands of concurrent requests)?
- SpecGuard uses "lightweight verifiers distilled from target hidden states" — what is the distillation cost and when must it be redone for a new target model?
- Does SpecuStream's adaptive K work well with P-EAGLE's single-pass drafting (where K is a training-time parameter, not runtime-tunable)?
- Does SMC-SD's approximate distribution (within 3% accuracy) degrade further for reasoning-heavy tasks that require exact token-level fidelity?
- At what number of particles N does SMC-SD achieve the 2.36× speedup over standard spec decode? Is there a throughput/quality tradeoff with N?
- Can SMC-SD be combined with P-EAGLE (use P-EAGLE for fast multi-token draft generation, SMC for better acceptance)?
- What are DiP-SD's quantitative speedups vs naive distributed SD and greedy batching? (arXiv 2604.20919 full PDF not yet fetched)
- Can DiP-SD's per-user adaptive K idea be adapted for vLLM's per-request K in co-located inference?
- How does DiP-SD perform when device draft models are heterogeneous (different architectures/sizes across users)?
