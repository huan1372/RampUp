---
title: "KV Cache Quantization"
tags: [quantization, kv-cache, memory, compression, turboquant]
created: 2026-04-16
updated: 2026-04-19
sources: [raw/vllm-releases.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md]
related: [concepts/kv-cache-management.md, techniques/fp8-quantization.md, techniques/prefix-caching.md]
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

### TurboQuant (as of April 19, 2026)
- Merged into vLLM main in PR #38479 (April 15, 2026)
- Refined in PR #40194 (April 17-19, 2026): removed redundant random sign flip from the WHT pipeline; this reduces per-token KV quantization overhead without affecting compression quality or preset behavior
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

## Open Questions

- Does TurboQuant work correctly with PagedAttention's block-based allocation? The PR targets V1 backend; V2/MRV2 compatibility is unconfirmed.
- What is the quality impact across the model zoo (not just Qwen3-4B)?
- Can the WHT rotation be fused into the attention kernel to eliminate overhead at short contexts?
- Is there a calibration-free method to select the optimal K/V bit allocation per model architecture?
- When will TurboQuant ship in a numbered vLLM release?

## Sources
- [raw/2026-04-16-turboquant-kv-compression-pr38479.md](../../raw/2026-04-16-turboquant-kv-compression-pr38479.md) — primary source for TurboQuant details, PR #38479
- [raw/2026-04-15-vllm-v019-release.md](../../raw/2026-04-15-vllm-v019-release.md) — context for FP8 KV and vLLM version state
- [raw/vllm-releases.md](../../raw/vllm-releases.md) — FP8 KV cache history
