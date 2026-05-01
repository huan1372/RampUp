---
title: "Overview & Synthesis"
tags: [overview, synthesis, meta]
created: 2026-04-14
updated: 2026-05-01
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-benchmarks-2026.md, raw/vllm-releases.md, raw/2026-04-14-vllm-rampup-recap.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md, raw/2026-04-19-calibrated-speculative-decoding-arxiv.md, raw/2026-04-20-specguard-arxiv-2604-15244.md, raw/2026-04-20-streamserve-arxiv-2604-09562.md, raw/2026-04-20-prefill-as-a-service-arxiv-2604-15039.md, raw/2026-04-21-vllm-v0191-release.md, raw/2026-04-21-yoco-plus-arxiv.md, raw/2026-04-21-fp16-kv-divergence-arxiv.md, raw/2026-04-22-vllm-prs-apr21-22.md, raw/2026-04-22-isoquant-arxiv.md, raw/2026-04-22-sequential-kv-trie-arxiv.md, raw/2026-04-23-vllm-prs-apr22-23.md, raw/2026-04-24-vllm-v020-release.md, raw/2026-04-24-deepseek-v4-vllm.md, raw/2026-04-24-vllm-prs-apr23-24.md, raw/2026-04-24-ttkv-arxiv.md, raw/2026-04-24-hybridgen-arxiv.md, raw/2026-04-24-smc-sd-arxiv.md, raw/2026-04-24-grace-kv-arxiv.md, raw/2026-04-24-realb-moe-arxiv.md, raw/2026-04-24-ragged-paged-attention-tpu-arxiv.md, raw/2026-04-25-vllm-prs-apr24-25.md, raw/2026-04-26-vllm-prs-apr25-26.md, raw/2026-04-26-dip-sd-arxiv-2604-20919.md, raw/2026-04-27-vllm-prs-apr26-27.md, raw/2026-04-28-vllm-prs-apr27-28.md, raw/2026-04-30-paypal-eagle3-production-arxiv-2604-19767.md, raw/2026-05-01-arxiv-2604-25975-capkv.md, raw/2026-05-01-arxiv-2604-26412-kvshot-speculative.md, raw/2026-05-01-vllm-prs-may1.md]
related: [concepts/paged-attention.md, concepts/model-runner-v2.md, concepts/continuous-batching.md, concepts/chunked-prefill.md, concepts/deepseek-v4-attention.md, techniques/cpu-gpu-hybrid-attention.md]
---

# Inference Optimization — Overview & Synthesis

## The Landscape (April 2026)

LLM inference optimization has converged on a core set of techniques that work together: efficient memory management (PagedAttention), dynamic scheduling (continuous batching), and hardware-aware computation (quantization, parallelism). The three dominant open-source serving engines are **vLLM**, **SGLang**, and **TensorRT-LLM**, each with different strengths.

## vLLM: Current State

vLLM (v0.20.0 as of April 23, 2026) has become the most widely adopted open-source inference engine. Key recent developments:

- **Model Runner V2 (MRV2)** — a ground-up rewrite of the execution core; opt-in in v0.18.0 (March 2026), **default in v0.19.0** (April 3, 2026). Cleaner, more modular, GPU-native input preparation, async-first. Delivers 56% throughput gain on GB200 from input prep alone.
- **V1 Engine** — the default since v0.8.0, delivering 1.7x throughput over the original engine. Prefix caching is now nearly free (<1% overhead at 0% hit rate).
- **Blackwell support** — full SM120 support as of v0.15.1 (Feb 2026), including NVFP4 MoE kernels.
- **Compilation** — moving toward `torch.compile` as the default optimization path, with custom Helion kernels planned.
- **v0.19.1** (April 18, 2026) — patch release upgrading to Transformers v5.5.3, unblocking Gemma4 from PyPI; adds Gemma4 Eagle3 speculative decoding (PR #39450) and quantized MoE for Gemma4 (PR #39045). (source: raw/2026-04-21-vllm-v0191-release.md)
- **v0.20.0** (April 23, 2026) — major release: 546 commits from 257 contributors. PyTorch 2.11 + CUDA 13.0 as new defaults; FA4 as default MLA prefill backend (SM90+); TurboQuant 2-bit KV (4× capacity); per-token-head INT8/FP8 KV quantization; online quantization frontend; vLLM IR skeleton; RayExecutorV2; 3FS KVConnector; AOT batch-invariance compile mode; CPU draft-model spec decode; full CUDA graph for Eagle prefill; MXFP4 W4A4 CUTLASS MoE for SM100; MXFP8 Marlin GEMM; NVFP4 dense on MI300/MI355X via emulation. (source: raw/2026-04-24-vllm-v020-release.md)

## Competitive Positioning

Based on Clarifai benchmarks (GPT-OSS-120B on 2x H100):
- **vLLM**: highest throughput at high concurrency (4,741 tok/s at 100 requests), fastest TTFT
- **SGLang**: most stable inter-token latency (4-21ms), strong RadixAttention for multi-turn
- **TensorRT-LLM**: best single-request throughput, but scales worse and requires compilation step

## Key Optimization Vectors

1. **Memory** — PagedAttention, KV cache offloading to CPU, FP8/FP4 quantization, sub-FP8 KV compression (TurboQuant: 2.6–4.9×, merged Apr 2026; WHT overhead reduced Apr 2026), cross-layer KV compression (YOCO++: 50% KV reduction via architecture, Apr 2026 research); CapKV information-theoretic eviction via leverage scores (outperforms H2O/SnapKV, Apr 28 2026 research)
2. **Compute** — speculative decoding (P-EAGLE 1.55–1.69×; CSD 2.33× peak, Apr 2026; Eagle3 + Gemma4 v0.19.1); FP8 per-token group quant packed kernel for Blackwell (PR #41326, May 1 2026); continuous batching, chunked prefill, fused kernels; MXFP4 W4A4 CUTLASS MoE kernel for B200 (Apr 2026)
3. **Scale** — tensor/pipeline/expert parallelism, disaggregated prefill-decode, elastic serving; FlashInfer FP8 async TP allreduce fusion (PR #39505, May 1 2026)
4. **Scheduling** — DBO (Dual-Batch Overlap), async scheduling with zero-bubble overlap; multimodal scheduler overhead reduction (Apr 2026); HMA KV offload scheduler SWA group support (PR #41228, May 1 2026)

### KV Cache Compression: Expanding Beyond FP8 (April 2026)

TurboQuant (PR #38479, merged April 15, 2026) extends vLLM's KV cache compression below FP8 for the first time. Using WHT rotation on keys and uniform quantization on values, it achieves 2.6–4.9× compression ratios at the cost of higher compute overhead and model-dependent quality risk. The conservative `turboquant_k8v4` preset (FP8 keys, 4-bit values) delivers TPOT improvement on long-context workloads with modest throughput overhead. Aggressive 3-bit compression shows severe quality degradation and requires validation. PR #40194 (April 2026) removes a redundant random sign flip from the WHT pipeline, reducing per-token overhead. See [KV Cache Quantization](techniques/kv-cache-quantization.md). (source: raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md)

### MXFP4 W4A4 for Blackwell MoE: Pushing to 4-Bit Activations (April 2026)

PR #37463 (merged April 2026) adds a CUTLASS-based W4A4 MXFP4 MoE kernel targeting SM100 (Blackwell B200). This is the first CUTLASS alternative to the FlashInfer MXFP4 MoE backend, exploiting SM100 native FP4 TensorCores and TMA async memory loads. W4A4 quantizes both weights and activations to MXFP4 format, maximizing MoE expert GEMM throughput on B200 at the cost of higher accuracy risk. Specific benchmark numbers vs FlashInfer not yet published. See [FP4 Quantization](techniques/fp4-quantization.md). (source: raw/2026-04-19-vllm-prs-apr17-19.md)

### Calibrated Speculative Decoding: Reducing False Rejections (April 2026)

arXiv 2604.13634 (April 15, 2026) proposes CSD, a training-free addition to speculative decoding that addresses false rejections — cases where a semantically valid draft token is discarded for being lexically different from the target distribution's top token. CSD adds Online Correction Memory (tracks historical rejections to propose rescue candidates) and Semantic Consistency Gating (validates candidates via probability ratio). Peak 2.33× throughput speedup. Not yet integrated into vLLM. See [Speculative Decoding](techniques/speculative-decoding.md). (source: raw/2026-04-19-calibrated-speculative-decoding-arxiv.md)

### KVShot: Why Draft Accuracy Decays at Long Speculation Steps (April 2026)

arXiv 2604.26412 (late April 2026) diagnoses a structural limitation of all hidden-state-based speculative decoding drafters (EAGLE, EAGLE-3, P-EAGLE): acceptance rate decays as speculation step index k increases. The root cause is that the target hidden state is a biased context compression — aligned to the current token position but missing the broader context needed at k > 3. The KVShot framework tests three drafter paradigms (hidden-only, KV-only, hybrid) on Qwen3-8B; KV-Reuse improves long-range acceptance but achieves only marginal end-to-end speedup due to two structural bottlenecks: shallow drafters cannot compute accurate key projections, and KV projection layers receive sparse gradient signals during training. Proposed fix: block-wise training. Not in vLLM. See [Speculative Decoding](techniques/speculative-decoding.md). (source: raw/2026-05-01-arxiv-2604-26412-kvshot-speculative.md)

### CapKV: Information-Theoretic KV Cache Eviction (April 2026)

arXiv 2604.25975 (April 28, 2026) replaces heuristic KV cache eviction (H2O, SnapKV, ScissorHands) with a theoretically grounded Information Bottleneck objective. Under a linear-Gaussian attention surrogate, the optimal retained KV subset maximizes mutual information with future attention outputs — approximated by log-determinant maximization via statistical leverage scores. Consistently outperforms all tested heuristic methods at high compression ratios on long-context benchmarks. Not in vLLM. See [KV Cache Management](concepts/kv-cache-management.md). (source: raw/2026-05-01-arxiv-2604-25975-capkv.md)

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

### vLLM v0.20.0: Major Release (April 23, 2026)

546 commits from 257 contributors. Key inference-optimization highlights:

- **FA4 as default MLA prefill backend** (SM90+, head-dim 512, paged-KV): corrects previous FA2 regression on Hopper/Blackwell MLA paths.
- **TurboQuant 2-bit KV** (4× capacity): extends TurboQuant family to 2-bit; first time TurboQuant ships in a numbered release (previously main-only since April 15).
- **Per-token-head INT8/FP8 KV**: finer-grained scale factors vs prior per-tensor quantization.
- **Online quantization frontend**: end-to-end quantization at load time without pre-quantized checkpoints.
- **vLLM IR**: initial `rms_norm` skeleton; foundation for future Helion kernel generation.
- **RayExecutorV2**: cleaner distributed execution backend.
- **3FS KVConnector**: file-system-backed KV sharing via Bytedance 3FS.
- **AOT batch-invariance compile mode**: compile artifacts shared across batch sizes.
- **CPU draft-model spec decode**: first time CPU-hosted draft models work in speculative pipeline.
- **CUDA graph Eagle prefill**: full graph capture for Eagle prefill paths.
- **MXFP4 W4A4 CUTLASS MoE (SM100)** and **MXFP8 Marlin GEMM**: Blackwell and Hopper quantization kernel additions.
- **DeepGEMM in wheel**: no separate install step.
- **Breaking**: PyTorch 2.11 + CUDA 13.0 required; metrics API changes; pooler config renames.

(source: raw/2026-04-24-vllm-v020-release.md)

### DeepSeek V4: Day-0 vLLM Support (April 24, 2026)

DeepSeek released V4-Pro (1.6T/49B active MoE) and V4-Flash (284B/13B active MoE) with 1M-token context, Apache 2.0 license. vLLM has day-0 support.

The architecture introduces a new hybrid attention paradigm:
- **CSA (Compressed Sparse Attention)**: compresses every m tokens into 1 KV entry; learned Lightning Indexer retrieves top-k relevant blocks; sliding window of recent uncompressed tokens added for full attention.
- **HCA (Heavily Compressed Attention)**: larger fixed compression group (m'=128); all compressed entries attended to; used for coarse-context layers.
- **mHC (Manifold-Constrained Hyper-Connections)**: residual connection constrained to Birkhoff polytope (doubly stochastic matrices), bounding spectral norm ≤ 1 — ensures numerical stability at 1M context depth.

Efficiency at 1M context: V4-Pro uses 27% of V3.2's FLOPs and 10% of V3.2's KV cache. V4-Flash: 10% of FLOPs, 7% of KV. This makes 1M-context serving economically comparable to shorter-context V3.2 serving.

This is the first production-grade open model achieving 1M context without linear attention. It signals that learned sparse attention (CSA/HCA style) is the next architectural layer above MLA/GQA. vLLM's implementation requires hybrid KV cache layout, kernel-fused Lightning Indexer retrieval, and disaggregated serving for V4-Pro scale. See [DeepSeek V4 Attention](concepts/deepseek-v4-attention.md).

(source: raw/2026-04-24-deepseek-v4-vllm.md)

### vLLM Mainline PRs: April 22–23, 2026

No new numbered release. Three performance-relevant PRs merged to main:

- **PR #40092 — TurboQuant FA3/FA4 Prefill Support** (April 23): Fixes TurboQuant's prefill paths to select FA3 on Hopper (H100/H200/H20, SM90) and FA4 on Blackwell (SM100) instead of defaulting to FA2. On H20: +71–89% throughput, −43–54% TTFT on prefill-heavy workloads. Also fixes mixed-backend failures for models with heterogeneous layer types.
- **PR #35737 — NVFP4 MoE Emulation Fallback** (April 22): Adds Triton-based software emulation for NVFP4-quantized MoE checkpoints on H100 (SM90) and AMD MI300/MI350. Previously Blackwell-only. Accuracy cost: ~3.3% word perplexity increase (10.90→11.26 on Qwen3-30B-A3B-NVFP4). Compute throughput lower than Blackwell native.
- **PR #40151 — FX Graph Deserialization Elimination** (April 23): Eliminates warm compile overhead by inlining attention submodules as Python code instead of deserializing serialized FX graphs. Warm compile times reduced 88–96% across major models (DeepSeek-V3.2: 6.05→0.27s; GPT-OSS-120B: 1.57→0.19s).

(source: raw/2026-04-23-vllm-prs-apr22-23.md)

### Deferred Papers Ingested (April 24, 2026 — Session 2)

Six arXiv papers that were inaccessible in prior sessions (HTTP 403) are now retrievable via search snippets.

**TTKV — Temporal KV Tiering (arXiv 2604.19769, Harbin IT + Guangzhou Univ):** HBM/DRAM tier assignment by temporal proximity; block-wise streaming attention for overlap. On 128K-context tasks: 5.94× traffic reduction, 76% latency reduction, 2× throughput over baselines. Not in vLLM. See [KV Cache Management](concepts/kv-cache-management.md).

**HybridGen — CPU-GPU Hybrid Attention (arXiv 2604.18529, April 20 2026):** CPU computes attention over CXL-DRAM KV while GPU handles HBM KV, results merged via log-sum-exp. 1.41×–3.2× vs 6 baselines on 3 models × 11 sizes × 3 GPU platforms. Full KV fidelity (no eviction). See [CPU-GPU Hybrid Attention](techniques/cpu-gpu-hybrid-attention.md).

**SMC-SD — Sequential Monte Carlo Spec Decode (arXiv 2604.15672, Cornell/MIT/ETH):** Replaces rejection sampling with importance-weighted particle resampling. No rollback; fixed-size verification. 2.36× over standard spec decode, 5.2× over autoregressive, within 3% accuracy. Not in vLLM. See [Speculative Decoding](techniques/speculative-decoding.md).

**GRACE — Graph-Guided Channel Elimination (arXiv 2604.16983, Harbin IT):** Channel pruning with inter-channel dependency graphs; 60% KV dimension reduction, negligible quality loss on LongBench. Orthogonal to quantization (could compose). Not in vLLM. See [KV Cache Quantization](techniques/kv-cache-quantization.md).

**ReaLB — Multimodal MoE EP Load Balancing (arXiv 2604.19503):** Dynamic per-EP-rank precision (FP4 for vision-heavy ranks) eliminates EP load imbalance with zero routing overhead. 1.29× layer speedup, ≤1.2% accuracy loss. Implemented in vLLM. See [Tensor Parallelism](techniques/tensor-parallelism.md).

**Ragged Paged Attention (arXiv 2604.15464, April 16 2026):** TPU-native PagedAttention kernel (Pallas/Mosaic). Fine-grained tiling + fused pipeline + distribution-aware compilation for decode/prefill/mixed modes. 86% MBU decode, 73% MFU prefill on TPU7x/Llama3-8B. 5× throughput improvement since vLLM-TPU integration (Feb 2025). Primary TPU backend in both vLLM and SGLang. See [PagedAttention](concepts/paged-attention.md).

### vLLM Post-v0.20.0 PRs: April 24–25, 2026

No new numbered release. Three technically significant PRs merged:

- **PR #40810 — EPLB replica selection bias fix**: Knuth multiplicative hash replaces the prior hash function, which silently collapsed all tokens to one replica when `top_k` was a multiple of the replica count (>90% imbalance). After: max/mean workload ratio 1.07 on Qwen3.5-A17B/8×B200. This is a **correctness fix masquerading as a performance issue** — EPLB was effectively disabled for common EP configurations. See [Tensor Parallelism](techniques/tensor-parallelism.md).

- **PR #34556 — Humming JIT quantization kernel**: JIT-compiled kernel library from inclusionAI supporting W{1–8}A{16/8/4} × {GPTQ, AWQ, FP8, MXFP4, NVFP4, BITNET}. On H20: ~142 vs ~90 TFLOPS for W8A16 vs Marlin (~1.58× faster). CUDA graph compatible. **Broadest weight-bit range (W1–W8) in any single vLLM backend** — adds path for sub-INT4 inference on Ampere/Hopper without new kernel contributions. See [FP8 Quantization](techniques/fp8-quantization.md).

- **PR #40412 — NIXL EP batched-expert consistency**: Fixes silent miscategorization of NIXL EP in fused MoE configuration, ensuring NIXL EP and DeepEP LL are treated equivalently for activation format, shared-expert handling, and FP4 selection. See [Disaggregated Serving](techniques/disaggregated-serving.md).

(source: raw/2026-04-25-vllm-prs-apr24-25.md)

### vLLM Post-v0.20.0 PRs: April 25–26, 2026

No new numbered release. Four PRs of technical significance merged:

- **PR #40893 (Apr 26) — FlashInfer NVLink MNNVL workspace sizing fix**: Both FlashInfer NVLink managers allocated workspace to the DP group size instead of the EP group size. When `dp_size != ep_size`, an assertion failure crashed initialization. Fix: use `self.cpu_group` (always the EP group on EP communicators). Validated on Kimi-K2.5-NVFP4 with TP=2, DP=4, EP. This configuration — high DP + EP + multi-node NVLink — is the standard Blackwell MoE scale-out pattern. See [Disaggregated Serving](techniques/disaggregated-serving.md).

- **PR #40865 (Apr 25) — MoE routed output unpadding fix**: MoE runner was unconditionally unpadding (truncating) routed output before shared-expert addition, crashing GPT-OSS 20B on B200 (non-contiguous tensor error). Fix: conditional on `has_shared_expert or routed_output_transform`. Preserves Nemotron-Nano-v3 behavior. See [Tensor Parallelism](techniques/tensor-parallelism.md).

- **PR #39403 (Apr 25) — HMA multi-group KV offload store**: Part 11 of the HMA KV offload series. Extends `build_connector_meta` to calculate and store KV blocks across multiple groups, enabling KV cache offloading for HMA-class architectures (prerequisite for DeepSeek V4-style multi-tier KV). See [KV Cache Management](concepts/kv-cache-management.md).

- **PR #40806 (Apr 26) — DSML streaming fix for DeepSeek V4/3.2**: DSML sentinel token `｜DSML｜` leaked into streamed content when the marker spanned chunk boundaries. Fix adds `_extract_content()` with partial tag overlap detection. Correctness fix for streaming tool calls on V4, V4-Flash, V3.2. See [DeepSeek V4 Attention](concepts/deepseek-v4-attention.md).

(source: raw/2026-04-26-vllm-prs-apr25-26.md)

### DiP-SD: Distributed Pipelined Speculative Decoding at the Edge (arXiv 2604.20919, April 2026)

Introduces speculative decoding to the **multi-user edge inference** setting where on-device small LLMs draft and an edge server verifies. Two novel parallelism dimensions: (1) device-level distributed drafting (N users draft simultaneously), (2) phase-level draft-verify pipelining (device drafting for batch i+1 overlaps server verification of batch i). Co-optimizes user-to-batch assignment, per-user K, and pipeline batch count. Claims to be the first SD scheme to exploit batch-level pipeline parallelism in the distributed draft-verify setting. Not integrated into vLLM; relevant to per-request K tuning research. See [Speculative Decoding](techniques/speculative-decoding.md).

(source: raw/2026-04-26-dip-sd-arxiv-2604-20919.md)

### vLLM Post-v0.20.0 PRs: April 26–27, 2026

No new numbered release. Five PRs of technical significance:

- **PR #40941 (Apr 27) — TurboQuant shared dequant buffers**: Per-layer dequant buffers consumed 57.6 GB at 1M tokens (32B model, TP=4); moved to global WorkspaceManager pool (one layer's worth reused). Memory savings: 472 MB at 8K, 7.4 GB at 128K, **57.6 GB at 1M**. Enables CUDA Graph capture for TurboQuant. No throughput change. Makes long-context TurboQuant inference practical. See [KV Cache Quantization](techniques/kv-cache-quantization.md).

- **PR #40950 (Apr 27) — DeepSeek V4 SiLU clamp for shared expert**: Numerical stability fix baking clamp limits into the `silu_and_mul` CUDA kernel for V4's `DeepseekV4MLP` shared expert. Prevents overflow at extreme activation magnitudes. See [DeepSeek V4 Attention](concepts/deepseek-v4-attention.md).

- **PR #38065 (Apr 27) — FP8 FlashInfer attention for ViT encoders**: FP8 quantization for Qwen3 VL family ViT encoder attention via cuDNN backend. On Qwen3-VL-30B / GB200: 1.08× at QHD, **1.18× at 4K** resolution. Core kernel 1.42× on GB300. Requires SM90+, cuDNN 9.17.1+. Benefit is resolution-dependent — HD resolution sees slight regression. See [FP8 Quantization](techniques/fp8-quantization.md).

- **PR #40946 (Apr 27) — SWA/chunked-local scheduler admission fix**: Fixes scheduler deadlock on hybrid SWA + full-attention models (Mistral, Gemma SWA variants). Root cause: startup pool sizer accounted for block recycling, runtime admission gate did not — deadlocking on long prompts. Unified `max_admission_blocks_per_request` method resolves the formula mismatch. See [KV Cache Management](concepts/kv-cache-management.md).

- **PR #40346 (Apr 26) — KV offload all blocks for remote decode in P/D**: Resets `start_block_idx` to 0 when `do_remote_decode=True`, exporting full KV state to CPU for cross-node decode. Prerequisite for CPU-backed P/D transport (complement to direct NVLink/RDMA paths). See [Disaggregated Serving](techniques/disaggregated-serving.md).

(source: raw/2026-04-27-vllm-prs-apr26-27.md)

### vLLM Post-v0.20.0 PRs: April 27–28, 2026

No new numbered release. Four PRs merged:

- **PR #40410 (Apr 27) — Eagle prefill metadata skip in MRV2**: Passes the target model's pre-built `CapturedAttentionState` (attention metadata + slot mappings) to `PrefillEagleCudaGraphManager` instead of rebuilding. Previously rebuilt three times per speculative step; now rebuilt once. ~5–10% end-to-end latency improvement for Eagle speculative decode. See [Model Runner V2](concepts/model-runner-v2.md), [Speculative Decoding](techniques/speculative-decoding.md).

- **PR #39930 (Apr 28) — Independent drafter attention backend selection**: New `--speculative-config.attention_backend` option breaks the forced drafter=target backend coupling. Drafter auto-selects independently when unspecified (not inherited). Resolves MLA-vs-GQA incompatibilities and non-causal attention failures (e.g., DFlash). See [Speculative Decoding](techniques/speculative-decoding.md).

- **PR #41049 (Apr 28) — StepPool None append fix for chunked prefill embeddings**: Fixes double-append bug in `StepPool.forward` that misaligned pooling output with batch requests for embedding models using long-sequence chunked prefill (4K+ tokens). Correctness fix; no throughput change. See [Chunked Prefill](concepts/chunked-prefill.md).

- **PR #39141 (Apr 27) — TRTLLM MoE routing method update**: Adds `SigmoidRenorm` and `MiniMax2` routing methods to TRT-LLM MoE backend, aligned with FlashInfer v0.6.8. Reclassifies `Custom`/`Simulated` as internal vLLM-specific. Performance improvement for MiniMax-series model serving via TRT-LLM backend. See [Tensor Parallelism](techniques/tensor-parallelism.md).

(source: raw/2026-04-28-vllm-prs-apr27-28.md)

### EAGLE3 in Production: PayPal Commerce Agent (arXiv 2604.19767, April 2026)

First published production deployment study of EAGLE3 speculative decoding via vLLM on a fine-tuned SLM (llama3.1-nemotron-nano-8B-v1, PayPal's commerce agent). Baseline: NVIDIA NIM on identical 2×H100 hardware.

- **gamma=3**: 22–49% throughput improvement and 18–33% latency reduction vs NIM; acceptance rate ~35.5% stable across concurrency 1–32 and temperature {0, 0.5}
- **GPU cost**: single H100 + EAGLE3 matches 2×H100 NIM → 50% hardware cost reduction
- **Quality**: LLM-as-Judge confirms no degradation

Key implication: on structured-output SLM workloads, EAGLE3 acceptance rates scale stably with concurrency (unlike the diminishing returns seen with P-EAGLE at high concurrency). Spec decode ROI is workload-dependent — commerce agents with predictable structure outperform open-ended generation settings.

(source: raw/2026-04-30-paypal-eagle3-production-arxiv-2604-19767.md)

## Open Questions

- How does MRV2 performance compare to MRV1 across the model zoo? (MRV1 still handles "long tail" cases)
- Cold compile times remain unaddressed — what are they after PR #40151, and do they matter for production restart SLOs?
- At what request concurrency does DeepSeek V4-Pro's CSA+HCA overhead break even with its 90% KV cache reduction? Where is the throughput crossover vs V3.2?
- Will vLLM's CSA+HCA implementation generalize to other models that adopt this architecture, or is it V4-specific?
- What is the quality degradation from NVFP4 emulation on AMD MI300/MI355X (added in v0.20.0) vs. the H100 emulation accuracy loss (3.3% perplexity) already characterized?
- How does vLLM's CPU KV cache offloading compare to SGLang's approach?
- When do hybrid-attention models with small KV cache become the mainstream serving target, shifting PD disaggregation economics?
- When will vLLM support YOCO-family cross-layer KV architectures?
- How should KV quantization quality benchmarks account for the FP16 baseline divergence (arXiv 2604.15409)?
- Does the depth–cache tradeoff (arXiv 2604.17935) hold empirically? At what KV compression ratio does reasoning degrade for current models?
- What does phase 2 of MLA + quantization fusion (issue #35792) involve beyond group FP8?
- What is the throughput overhead of NVFP4 emulation on H100 relative to BF16 baseline? (memory savings vs compute cost tradeoff)
- Can IsoQuant be fused with the attention kernel, similar to MLA + FP8 fusion (PR #38877)?
- How do TTKV (temporal tiering) and HybridGen (semantic-aware CPU-GPU split) compare on the same 128K+ context benchmark? Can they compose?
- Does SMC-SD's approximate distribution cause measurable degradation on reasoning-heavy tasks beyond the 3% accuracy bound?
- Can GRACE (channel elimination) and TurboQuant (bit reduction) be composed for >5× KV compression at acceptable quality?
- Does ReaLB's per-rank FP4 work on SM90 (H100) via MXFP8 fallback, or is it Blackwell-only?
- RPA achieves 86% MBU on TPU7x for Llama3-8B — how does this compare to H100 FlashAttention MBU for the same model?
