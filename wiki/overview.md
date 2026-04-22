---
title: "Overview & Synthesis"
tags: [overview, synthesis, meta]
created: 2026-04-14
updated: 2026-04-22
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-benchmarks-2026.md, raw/vllm-releases.md, raw/2026-04-14-vllm-rampup-recap.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md, raw/2026-04-19-calibrated-speculative-decoding-arxiv.md, raw/2026-04-20-specguard-arxiv-2604-15244.md, raw/2026-04-20-streamserve-arxiv-2604-09562.md, raw/2026-04-20-prefill-as-a-service-arxiv-2604-15039.md, raw/2026-04-21-vllm-v0191-release.md, raw/2026-04-21-yoco-plus-arxiv.md, raw/2026-04-21-fp16-kv-divergence-arxiv.md, raw/2026-04-22-vllm-prs-apr21-22.md, raw/2026-04-22-isoquant-arxiv.md, raw/2026-04-22-sequential-kv-trie-arxiv.md]
related: [concepts/paged-attention.md, concepts/model-runner-v2.md, concepts/continuous-batching.md, concepts/chunked-prefill.md]
---

# Inference Optimization — Overview & Synthesis

## The Landscape (April 2026)

LLM inference optimization has converged on a core set of techniques that work together: efficient memory management (PagedAttention), dynamic scheduling (continuous batching), and hardware-aware computation (quantization, parallelism). The three dominant open-source serving engines are **vLLM**, **SGLang**, and **TensorRT-LLM**, each with different strengths.

## vLLM: Current State

vLLM (v0.19.1 as of April 18, 2026) has become the most widely adopted open-source inference engine. Key recent developments:

- **Model Runner V2 (MRV2)** — a ground-up rewrite of the execution core; opt-in in v0.18.0 (March 2026), **default in v0.19.0** (April 3, 2026). Cleaner, more modular, GPU-native input preparation, async-first. Delivers 56% throughput gain on GB200 from input prep alone.
- **V1 Engine** — the default since v0.8.0, delivering 1.7x throughput over the original engine. Prefix caching is now nearly free (<1% overhead at 0% hit rate).
- **Blackwell support** — full SM120 support as of v0.15.1 (Feb 2026), including NVFP4 MoE kernels.
- **Compilation** — moving toward `torch.compile` as the default optimization path, with custom Helion kernels planned.
- **v0.19.1** (April 18, 2026) — patch release upgrading to Transformers v5.5.3, unblocking Gemma4 from PyPI; adds Gemma4 Eagle3 speculative decoding (PR #39450) and quantized MoE for Gemma4 (PR #39045). (source: raw/2026-04-21-vllm-v0191-release.md)

## Competitive Positioning

Based on Clarifai benchmarks (GPT-OSS-120B on 2x H100):
- **vLLM**: highest throughput at high concurrency (4,741 tok/s at 100 requests), fastest TTFT
- **SGLang**: most stable inter-token latency (4-21ms), strong RadixAttention for multi-turn
- **TensorRT-LLM**: best single-request throughput, but scales worse and requires compilation step

## Key Optimization Vectors

1. **Memory** — PagedAttention, KV cache offloading to CPU, FP8/FP4 quantization, sub-FP8 KV compression (TurboQuant: 2.6–4.9×, merged Apr 2026; WHT overhead reduced Apr 2026), cross-layer KV compression (YOCO++: 50% KV reduction via architecture, Apr 2026 research)
2. **Compute** — speculative decoding (P-EAGLE 1.55–1.69×; CSD 2.33× peak, Apr 2026; Eagle3 + Gemma4 v0.19.1), continuous batching, chunked prefill, fused kernels; MXFP4 W4A4 CUTLASS MoE kernel for B200 (Apr 2026)
3. **Scale** — tensor/pipeline/expert parallelism, disaggregated prefill-decode, elastic serving
4. **Scheduling** — DBO (Dual-Batch Overlap), async scheduling with zero-bubble overlap; multimodal scheduler overhead reduction (Apr 2026)

### KV Cache Compression: Expanding Beyond FP8 (April 2026)

TurboQuant (PR #38479, merged April 15, 2026) extends vLLM's KV cache compression below FP8 for the first time. Using WHT rotation on keys and uniform quantization on values, it achieves 2.6–4.9× compression ratios at the cost of higher compute overhead and model-dependent quality risk. The conservative `turboquant_k8v4` preset (FP8 keys, 4-bit values) delivers TPOT improvement on long-context workloads with modest throughput overhead. Aggressive 3-bit compression shows severe quality degradation and requires validation. PR #40194 (April 2026) removes a redundant random sign flip from the WHT pipeline, reducing per-token overhead. See [KV Cache Quantization](techniques/kv-cache-quantization.md). (source: raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md)

### MXFP4 W4A4 for Blackwell MoE: Pushing to 4-Bit Activations (April 2026)

PR #37463 (merged April 2026) adds a CUTLASS-based W4A4 MXFP4 MoE kernel targeting SM100 (Blackwell B200). This is the first CUTLASS alternative to the FlashInfer MXFP4 MoE backend, exploiting SM100 native FP4 TensorCores and TMA async memory loads. W4A4 quantizes both weights and activations to MXFP4 format, maximizing MoE expert GEMM throughput on B200 at the cost of higher accuracy risk. Specific benchmark numbers vs FlashInfer not yet published. See [FP4 Quantization](techniques/fp4-quantization.md). (source: raw/2026-04-19-vllm-prs-apr17-19.md)

### Calibrated Speculative Decoding: Reducing False Rejections (April 2026)

arXiv 2604.13634 (April 15, 2026) proposes CSD, a training-free addition to speculative decoding that addresses false rejections — cases where a semantically valid draft token is discarded for being lexically different from the target distribution's top token. CSD adds Online Correction Memory (tracks historical rejections to propose rescue candidates) and Semantic Consistency Gating (validates candidates via probability ratio). Peak 2.33× throughput speedup. Not yet integrated into vLLM. See [Speculative Decoding](techniques/speculative-decoding.md). (source: raw/2026-04-19-calibrated-speculative-decoding-arxiv.md)

## Glossary (quick reference)

- **TTFT** — Time To First Token. Latency from prompt submission to first response token. Measures prefill speed. Target: <200ms.
- **ITL** — Inter-Token Latency. Time between consecutive output tokens during streaming. Measures decode speed. Target: <30ms.
- **KV cache** — Stored key-value tensors from attention for all tokens seen so far. See [KV Cache Management](concepts/kv-cache-management.md).
- **Prefill** — Processing input prompt (parallel, compute-bound).
- **Decode** — Generating output tokens one at a time (sequential, memory-bandwidth-bound).
- **Preemption** — Evicting a request's KV blocks when memory runs out (swap to CPU or recompute). See [PagedAttention](concepts/paged-attention.md).
- **TP** — Tensor Parallelism. Splitting model layers across multiple GPUs. See [Tensor Parallelism](techniques/tensor-parallelism.md).

(source: raw/2026-04-14-vllm-rampup-recap.md)

### SpecGuard: Step-Level Verification for Reasoning (April 2026)

arXiv 2604.15244 proposes SpecGuard, which extends speculative decoding with step-level verification for multi-step reasoning. Standard spec decode is token-centric — erroneous reasoning steps are accepted and propagate forward. SpecGuard uses model-internal verifier ensembles (no external reward model) to validate each reasoning step and recompute only erroneous steps. Results: +3.6% accuracy, −11% latency vs standard SD. Not yet in vLLM. See [Speculative Decoding](techniques/speculative-decoding.md). (source: raw/2026-04-20-specguard-arxiv-2604-15244.md)

### StreamServe: Disaggregated Serving + Adaptive Speculation (April 2026)

arXiv 2604.09562 combines disaggregated P/D (PipeServe-Engine) with runtime-adaptive speculation depth (SpecuStream). Key insight: static speculation depth K is suboptimal for reasoning tasks where acceptance rates fluctuate mid-sequence. SpecuStream tunes K online from live acceptance signals. On GSM8K: 264 vs 241 tok/s (+9.5% throughput), 0.30 vs 3.50 s (−91% latency) vs TP-vLLM baseline. Overall 11–18× latency reduction across benchmarks. Evaluated on A800 GPUs, small scale (320 queries). See [Disaggregated Serving](techniques/disaggregated-serving.md). (source: raw/2026-04-20-streamserve-arxiv-2604-09562.md)

### Prefill-as-a-Service: Cross-Datacenter Disaggregation via Hybrid Attention (April 2026)

arXiv 2604.15039 (Moonshot AI / Kimi) argues that hybrid-attention architectures produce small enough KV caches to enable cross-datacenter prefill-decode disaggregation over commodity Ethernet. PrfaaS adds selective offloading, bandwidth-aware scheduling, and cache-aware placement on top of model-side KV efficiency. On an internal 1T-parameter hybrid model: +54% throughput vs homogeneous PD, +32% vs naive heterogeneous. Key implication: as hybrid models become dominant, the intra-DC coupling assumption of PD disaggregation breaks down. See [Disaggregated Serving](techniques/disaggregated-serving.md). (source: raw/2026-04-20-prefill-as-a-service-arxiv-2604-15039.md)

### FP16 KV Cache: Numerical Non-Equivalence (April 2026)

arXiv 2604.15409 (Chodavarapu, Xu) establishes that FP16 KV-cached inference deterministically diverges from cache-free recomputation due to FP16 non-associativity. 100% token divergence rate across LLaMA-2-7B, Mistral-7B-v0.3, Gemma-2-2B on GSM8K — but cache-ON yields higher accuracy in 8 of 9 conditions. FP32 eliminates divergence entirely. This result undermines the universally assumed numerical equivalence of KV caching, with implications for correctness testing, reproducibility, and quantization quality evaluations. See [KV Cache Management](concepts/kv-cache-management.md). (source: raw/2026-04-21-fp16-kv-divergence-arxiv.md)

### Cross-Layer KV Compression: YOCO++ (April 2026)

arXiv 2604.13556 introduces YOCO++ — a cross-layer KV sharing architecture that achieves 50% KV cache reduction by sharing global KV tensors across top-half Transformer layers with learned residual connections. Achieves SOTA quality among cross-layer methods at 50% compression. Not yet supported in vLLM or other production serving systems. See [Cross-Layer KV Compression](techniques/cross-layer-kv-compression.md). (source: raw/2026-04-21-yoco-plus-arxiv.md)

### vLLM Mainline PRs: April 21–22, 2026

No new numbered release. Four PRs merged to main:

- **PR #40413 — Fused RMS Norm Batch Invariant** (April 21): Removes redundant conditional check in `fused_add_rms_norm`; 2.1% E2E latency improvement on Llama-3.1-8B-FP8. Affects all models using fused RMS norm (most modern LLMs).
- **PR #38284 — CUDAGraph Memory Profiling Default** (April 21): Enables CUDA graph memory profiling by default; default `--gpu-memory-utilization` raised from 0.9 → **0.92**. Fixes startup OOM for DeepSeek-R1 at DP=8, EP configurations.
- **PR #38877 — MLA + Group FP8 Fusion** (April 22): Fuses per-group FP8 quantization into MLA attention kernel for DeepSeek V3. B200x4: 5,345 → 5,367 tok/s. Phase 1 of issue #35792.
- **PR #38453 — KV Offload HMA Multi-group** (April 22): Extends CPU-GPU KV offloading to support multi-group KV caches (needed for MLA-based models). No benchmark numbers; correctness fix.

(source: raw/2026-04-22-vllm-prs-apr21-22.md)

### Rotation-Family KV Compression: IsoQuant (arXiv 2603.28430, March 28, 2026)

A missed paper from March 28 now ingested. Extends TurboQuant's rotation pre-processing to SO(4) isoclinic decomposition: any SO(4) rotation = two quaternion multiplications `T(v) = q_L ⊗ v ⊗ q̄_R`. Power-of-2 block size (4D) aligns with GPU SIMD. At d=128, IsoQuant-Full uses 1,024 FMAs vs RotorQuant's ~2,408 — achieving **4.49× speedup** over RotorQuant with identical reconstruction quality. IsoQuant-Fast (single isoclinic factor): 4.66× speedup.

Lineage: TurboQuant (dense d×d) → RotorQuant (Clifford rotors) → IsoQuant (SO(4) isoclinic). None of these rotation variants are yet integrated into vLLM.

(source: raw/2026-04-22-isoquant-arxiv.md)

### Sequential KV Compression: Beyond Per-Vector Shannon Limit (arXiv 2604.15356, April 2026)

A theoretical framework showing that all per-vector KV compression methods (FP8, TurboQuant, IsoQuant) are bounded by per-vector Shannon entropy, and that this bound can be broken by exploiting sequential structure. Uses two layers: (1) PLT-based semantic prefix deduplication across sessions (generalizing vLLM's prefix caching), (2) predictive delta coding using the model's own KV predictions. Compression ratio improves with context length, unlike fixed-ratio per-vector methods. No implementation or benchmark numbers. Theoretical alignment with prefix caching: if vLLM adds fuzzy prefix matching, Layer 1 becomes directly applicable.

(source: raw/2026-04-22-sequential-kv-trie-arxiv.md)

## Open Questions

- How does MRV2 performance compare to MRV1 across the model zoo? (MRV1 still handles "long tail" cases)
- What's the practical impact of torch.compile on cold-start times?
- How does vLLM's CPU KV cache offloading compare to SGLang's approach?
- When do hybrid-attention models with small KV cache become the mainstream serving target, shifting PD disaggregation economics?
- When will vLLM support YOCO-family cross-layer KV architectures?
- How should KV quantization quality benchmarks account for the FP16 baseline divergence (arXiv 2604.15409)?
- Does the depth–cache tradeoff (arXiv 2604.17935) hold empirically? At what KV compression ratio does reasoning degrade for current models?
- What does phase 2 of MLA + quantization fusion (issue #35792) involve beyond group FP8?
- Can IsoQuant be fused with the attention kernel, similar to MLA + FP8 fusion (PR #38877)?
