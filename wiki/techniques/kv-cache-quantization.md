---
title: "KV Cache Quantization"
tags: [quantization, kv-cache, memory, compression, turboquant, isoquant, sequential-compression, flashattention]
created: 2026-04-16
updated: 2026-04-23
sources: [raw/vllm-releases.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md, raw/2026-04-21-fp16-kv-divergence-arxiv.md, raw/2026-04-22-isoquant-arxiv.md, raw/2026-04-22-sequential-kv-trie-arxiv.md, raw/2026-04-23-vllm-prs-apr22-23.md]
related: [concepts/kv-cache-management.md, techniques/fp8-quantization.md, techniques/prefix-caching.md, techniques/cross-layer-kv-compression.md]
---

# KV Cache Quantization

## Summary

KV cache quantization compresses the key-value tensors stored during attention to reduce GPU memory usage, enabling larger batch sizes or longer contexts on fixed hardware. Unlike weight quantization (which is done once offline), KV cache quantization happens online during serving as tokens are generated. The primary trade-off is compression ratio vs. quality degradation vs. throughput overhead.

## Problem It Solves

KV cache memory scales linearly with sequence length × batch size × number of layers × number of heads × head dimension. For large models at high concurrency, the KV cache — not the model weights — becomes the dominant memory consumer. Reducing KV cache size directly increases the number of concurrent sequences that fit in GPU memory, improving throughput.

## The Compression Spectrum

| Method | KV bits | Compression vs BF16 | Quality risk | Throughput overhead | vLLM flag |
|--------|---------|---------------------|--------------|---------------------|-----------|
| BF16 (baseline) | 16 | 1× | None | None | (default) |
| FP8 KV | 8 | 2× | Minimal | Minimal (hardware-native) | `--kv-cache-dtype fp8` |
| TurboQuant k8v4 | 8K / 4V | 2.6× | Low | ~0–21% | `--kv-cache-dtype turboquant_k8v4` |
| TurboQuant 4-bit | 4 | 3.8× | Moderate | Not yet benchmarked at scale | `--kv-cache-dtype turboquant_4bit_nc` |
| TurboQuant 3-bit | 3 | 4.9× | High (0% GSM8K reported) | Not yet benchmarked at scale | `--kv-cache-dtype turboquant_3bit_nc` |

## How It Works

### FP8 KV Cache (production-ready)

Cast KV tensors from BF16 to FP8 as they are written to the cache. On H100/H200/B200 hardware with native FP8 tensor cores, dequantization is essentially free. Halves KV memory with near-zero quality impact.

(source: raw/vllm-releases.md, raw/2026-04-15-vllm-v019-release.md)

### TurboQuant (merged April 15, 2026 — vLLM main, PR #38479)

A sub-4-bit online KV compression approach using asymmetric treatment of keys and values:

**Keys (WHT + Lloyd-Max)**:
1. Apply Walsh-Hadamard Transform (WHT) rotation to decorrelate channels — spreads outlier values across all dimensions
2. Apply Lloyd-Max scalar quantization at 3–4 bits — optimal MSE scalar quantizer for a given bit budget
3. Optionally apply norm correction (`_nc` presets) to compensate for quantization error

**Values (uniform quantization)**:
- Empirically more uniform distribution than keys → direct uniform quantization at 2–4 bits without rotation

**Asymmetric K/V allocation** (key practical finding): allocating more bits to keys than values (`tq_k4v3`, `turboquant_k8v4`) consistently outperforms symmetric assignments. Symmetric 3-bit reportedly produces 0% GSM8K on some models.

(source: raw/2026-04-16-turboquant-kv-compression-pr38479.md)

## Implementation in vLLM

### FP8 KV Cache
- Available since vLLM v0.4 era; mature and stable
- Flag: `--kv-cache-dtype fp8`
- No offline preparation required
- Hardware-accelerated on H100, H200, B200; software fallback on older GPUs

### TurboQuant (as of April 23, 2026)
- Merged into vLLM main in PR #38479 (April 15, 2026)
- Refined in PR #40194 (April 17-19, 2026): removed redundant random sign flip from the WHT pipeline; reduces per-token KV quantization overhead without affecting quality
- **FA3/FA4 prefill support (PR #40092, April 23, 2026):** TurboQuant's prefill paths previously defaulted to FA2 on all hardware. PR #40092 adds FA version detection: FA3 is selected on SM90 (Hopper: H100, H200, H20), FA4 on SM100 (Blackwell). Also fixes mixed-backend failures when a model routes some layers to TurboQuant and others to standard FlashAttention. Unlocks the following throughput improvements on NVIDIA H20 (SM90):

  | Workload | Throughput improvement | TTFT reduction |
  |----------|------------------------|----------------|
  | prefill_heavy | +71–89% | −43–54% |
  | long_balanced | +46–58% | −62–63% |

  (source: raw/2026-04-23-vllm-prs-apr22-23.md)

- Targets vLLM V1 attention backend only
- Flag: `--kv-cache-dtype <preset>` where preset is one of: `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_3bit_nc`, `tq_k4v3`
- No offline calibration required — online compression at inference time
- FP8 fallback for CUDA architectures without WHT kernel support
- **Not yet in a numbered release** — available on vLLM main as of 2026-04-15

## Benchmarks

### TurboQuant k8v4 (Qwen3-4B, RTX PRO 6000 Blackwell)
| Metric | Baseline (BF16) | turboquant_k8v4 |
|--------|-----------------|-----------------|
| Compression ratio | 1× | 2.6× |
| Throughput | 100% | 79–100% |
| TPOT @ 8K context | 138.1 ms | 135.2 ms (faster) |

The TPOT improvement at 8K context arises because higher compression fits more sequences in-cache, improving batching efficiency. Throughput overhead dominates at shorter contexts.

(source: raw/2026-04-16-turboquant-kv-compression-pr38479.md)

## Trade-offs

**FP8 KV cache**:
- Gain: 2× KV memory reduction, essentially free on modern hardware
- Lose: nothing measurable in practice

**TurboQuant k8v4**:
- Gain: 2.6× KV compression, TPOT improvement on long-context workloads
- Lose: 0–21% throughput at short contexts; quality risk is low but model-dependent

**TurboQuant 4-bit / 3-bit**:
- Gain: 3.8–4.9× KV compression
- Lose: significant quality risk (3-bit showed 0% GSM8K on some configs); throughput impact not yet fully characterized

## When to Use

- **FP8 KV cache**: always enable on H100/H200/B200 — it is essentially free
- **TurboQuant k8v4**: consider for long-context workloads (>4K tokens) where TPOT improvement outweighs short-context overhead; validate quality on your model before production
- **TurboQuant 3/4-bit**: experimental — require thorough quality benchmarking; currently not recommended for production without validation
- **All sub-FP8 methods**: avoid with hybrid models (e.g., Qwen3.5 series) until compatibility is confirmed

## Numerical Precision Note: FP16 Baseline Divergence

KV quantization adds numerical error on top of a **pre-existing FP16 baseline divergence**. As of arXiv 2604.15409 (April 2026), FP16 KV-cached inference already diverges 100% token-by-token from cache-free recomputation even without quantization, due to FP16 non-associativity. This means:

- Quality evaluations of FP8/TurboQuant that compare against a "BF16 baseline" are measuring divergence from BF16 — which is already diverging from theoretical exact computation
- The "FP8 KV is near-zero quality impact" claim is empirically valid but means: FP8's additional divergence over BF16 is small, not that either is zero-divergence from ground truth
- Research papers measuring quality degradation from KV quantization should ideally control for this baseline FP16 divergence

(source: raw/2026-04-21-fp16-kv-divergence-arxiv.md)

## Rotation-Family KV Compression (Research, April 2026)

TurboQuant's WHT rotation inspired a family of lighter-weight orthogonal rotation methods:

### RotorQuant (2026, not peer-reviewed)

Replaces TurboQuant's dense d×d rotation matrix with Clifford rotors in Cl(3,0). The rotor sandwich product `R x R̃` uses only ~100 multiply-adds per vector (vs. 16,384 for d=128), exploiting algebraic sparsity.

Performance vs TurboQuant: 10–19× faster on NVIDIA (CUDA), 9–31× on Apple Silicon (Metal), 44× fewer parameters. Attention cosine similarity 0.990 vs TurboQuant's 0.991 — essentially identical quality.

No vLLM integration; vLLM feature request in issue #38291, not merged.

(source: raw/2026-04-22-isoquant-arxiv.md)

### IsoQuant (arXiv 2603.28430, March 28, 2026)

Identifies that RotorQuant's Cl(3,0) partition is poorly hardware-aligned (non-power-of-2 block sizes). Replaces it with isoclinic decomposition of SO(4): any SO(4) rotation = left-isoclinic × right-isoclinic factor. In quaternion notation: `T(v) = q_L ⊗ v ⊗ q̄_R`.

Benefits: 4D quaternion blocks align with SIMD width and enable coalesced memory access.

| Variant | FMAs at d=128 | vs RotorQuant |
|---------|---------------|---------------|
| IsoQuant-Full | 1,024 | 4.49× faster |
| IsoQuant-Fast | 512 | 4.66× faster |

Quality: essentially identical to RotorQuant/TurboQuant across all 18 tested settings.

Compatible with any stage-2 quantizer (e.g., Lloyd-Max). No vLLM integration.

**Lineage:** TurboQuant (dense d×d) → RotorQuant (Clifford rotors) → IsoQuant (SO(4) isoclinic)

(source: raw/2026-04-22-isoquant-arxiv.md)

## Sequential KV Cache Compression (arXiv 2604.15356, April 2026)

All per-vector methods (FP8, TurboQuant, IsoQuant) are bounded by the Shannon entropy of individual KV vectors. This paper argues the KV cache is compressible **below** this per-vector limit by treating it as a sequence.

**Key insight:** KV vectors are samples from the model's own training language — the model is near-optimal at predicting what each successive KV will be. Storing only the residual `KV_t − model_prediction(KV_t)` can be far smaller than storing `KV_t` directly.

### Two-layer architecture:

**Layer 1 — Probabilistic prefix deduplication:** Uses Probabilistic Language Tries (PLT) to find semantically equivalent shared prefixes across sessions. Generalizes vLLM's exact-match prefix caching to approximate semantic matching, deduplicating shared KV state across sessions.

**Layer 2 — Predictive delta coding:** For each remaining position, store only `KV_t − f(context_{<t})`, where `f(·)` is the model's predicted KV. As context grows, the model's predictive distribution tightens and residuals shrink.

**Scaling property:** Compression ratio improves with context length — unlike per-vector methods which achieve fixed ratios regardless of context.

**Relationship to prefix caching:** Layer 1 is a strict generalization of vLLM's prefix caching. If vLLM eventually supports fuzzy prefix matching, Layer 1 becomes directly applicable.

**Limitations:** No benchmark numbers vs TurboQuant/FP8 in available content. PLT matching cost and delta coding throughput impact uncharacterized. No implementation or vLLM integration.

(source: raw/2026-04-22-sequential-kv-trie-arxiv.md)

## Open Questions

- Does TurboQuant work correctly with PagedAttention's block-based allocation? The PR targets V1 backend; V2/MRV2 compatibility is unconfirmed.
- What is the quality impact across the model zoo (not just Qwen3-4B)?
- Can the WHT rotation be fused into the attention kernel to eliminate overhead at short contexts?
- Is there a calibration-free method to select the optimal K/V bit allocation per model architecture?
- When will TurboQuant ship in a numbered vLLM release?
- How does the FP16 baseline divergence (arXiv 2604.15409) affect quality measurements for TurboQuant and future KV quantization methods?
- Can IsoQuant's quaternion rotation be fused with the attention kernel (like MLA+FP8 fusion in PR #38877)?
- What are the actual benchmark numbers for sequential compression (arXiv 2604.15356) vs FP8 and TurboQuant?
- Does the FA3/FA4 prefill fix (PR #40092) apply to all TurboQuant presets or only to `turboquant_k8v4`? What are the per-preset throughput numbers?

## Sources
- [raw/2026-04-16-turboquant-kv-compression-pr38479.md](../../raw/2026-04-16-turboquant-kv-compression-pr38479.md) — primary source for TurboQuant details, PR #38479
- [raw/2026-04-15-vllm-v019-release.md](../../raw/2026-04-15-vllm-v019-release.md) — context for FP8 KV and vLLM version state
- [raw/vllm-releases.md](../../raw/vllm-releases.md) — FP8 KV cache history
- [raw/2026-04-22-isoquant-arxiv.md](../../raw/2026-04-22-isoquant-arxiv.md) — IsoQuant and RotorQuant rotation family
- [raw/2026-04-22-sequential-kv-trie-arxiv.md](../../raw/2026-04-22-sequential-kv-trie-arxiv.md) — sequential compression beyond per-vector Shannon limit
- [raw/2026-04-23-vllm-prs-apr22-23.md](../../raw/2026-04-23-vllm-prs-apr22-23.md) — PR #40092 TurboQuant FA3/FA4 prefill support
