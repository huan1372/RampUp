---
title: "TurboQuant: 2-bit KV Cache Compression merged into vLLM main (PR #38479)"
source_type: github-pr
source_url: https://github.com/vllm-project/vllm/pull/38479
collected: 2026-04-16
tags: [kv-cache, quantization, memory, vllm, turboquant, compression]
---

# TurboQuant: 2-bit KV Cache Compression — vLLM PR #38479

**Merged**: April 15, 2026 into `vllm-project/vllm` main branch  
**Author**: vibhavagarwal5  
**Related (closed) PR**: #38280 (earlier TurboQuant implementation by lishunyang12, closed in favor of #38479)

---

## What It Does

Adds online KV cache compression to vLLM's **V1 attention backend** without requiring offline calibration or model weight modifications. Compression happens at serving time during the attention write step.

---

## Quantization Method

The system uses asymmetric treatment of keys and values:

**Keys** — WHT (Walsh-Hadamard Transform) rotation + Lloyd-Max scalar quantization:
1. Apply WHT rotation to decorrelate key channels
2. Quantize to 3–4 bits using Lloyd-Max optimal scalar quantization
3. WHT spreads outliers across all channels, making uniform scalar quantization viable

**Values** — Uniform quantization to 2–4 bits (no rotation needed; value vectors are empirically more uniform than keys).

**FP8 fallback** — used for older CUDA architectures that cannot support the WHT kernel path.

### Supported Presets

| Preset | Key bits | Value bits | Compression ratio |
|--------|----------|------------|-------------------|
| `turboquant_k8v4` | FP8 (8) | 4 | 2.6× |
| `turboquant_4bit_nc` | 4 (MSE + norm correction) | 4 | 3.8× |
| `turboquant_3bit_nc` | 3 (MSE + norm correction) | 3 | 4.9× |
| `tq_k4v3` | 4 | 3 | asymmetric; not in paper |

The `_nc` suffix denotes "norm correction" to compensate for quantization error.

---

## Predecessor PR (#38280) — Closed

PR #38280 implemented TurboQuant from the ICLR 2026 Google Research paper:
- Hadamard rotation + Lloyd-Max scalar quantization + outlier-aware channel allocation
- Compressed bf16 → packed 4-bit uint8 (~95 bytes/token vs 256 bytes baseline)
- Claimed 2.7× KV cache capacity on Qwen2.5-7B
- **Problems**: 0.36× baseline throughput (decompression cost dominated), 5.9× CUDA graph memory overhead, incompatible with hybrid models, incomplete QJL residual correction
- Contributor closed in favor of #38479

---

## Benchmark Data (PR #38479)

**Hardware**: RTX PRO 6000 Blackwell  
**Model**: Qwen3-4B

| Config | Throughput vs baseline | TPOT (8K context) |
|--------|------------------------|-------------------|
| Baseline (BF16 KV) | 100% | 138.1 ms |
| `turboquant_k8v4` | 79–100% | 135.2 ms (faster) |

- Long-sequence advantage: TPOT improves on long sequences because higher compression → more sequences fit in KV cache → better batching.
- Short-sequence overhead: quantization/dequantization adds latency at short contexts.

---

## Known Quality Issues

Community testing identified significant quality degradation at aggressive bit-widths:
- **Symmetric 3-bit**: 0% GSM8K accuracy on initial testing
- **Asymmetric K/V allocation** (e.g., more bits for keys than values) consistently outperforms symmetric approaches
- Quality highly model-dependent; Qwen3.5 hybrid models show more degradation than dense models

---

## Usage

Set `--kv-cache-dtype turboquant_k8v4` (or other preset) at vLLM serving time. No offline weight conversion required.

---

## Relation to Existing Work

- **FP8 KV cache** (`--kv-cache-dtype fp8`): existing 2× compression; native hardware acceleration on H100/H200/B200; no quality issues. TurboQuant targets 2.6–4.9× at the cost of higher compute overhead and model-dependent quality risk.
- **Original TurboQuant paper** (ICLR 2026, Google Research): describes offline/static compression; vLLM's implementation is an online adaptation.
- **QuantSpec** (arXiv 2502.10424): related work using hierarchical 4-bit quantized KV cache for self-speculative decoding.

---

## Notes for Ingest

- This technique is categorized as a **KV cache quantization** technique, distinct from weight quantization.
- Should create a new `wiki/techniques/kv-cache-quantization.md` page that covers the spectrum: FP8 (existing) → TurboQuant 4-bit → TurboQuant 3-bit → future lower-bit approaches.
- The asymmetric K/V quality finding is practically important and should be highlighted.
