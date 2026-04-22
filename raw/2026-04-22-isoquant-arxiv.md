---
title: "IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2603.28430
collected: 2026-04-22
tags: [kv-cache, compression, quantization, rotation, isoclinic, so4, turboquant, rotorquant]
---

# IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression

**arXiv:** 2603.28430
**Submitted:** March 28, 2026
**Authors:** (ParaMind2025 team — names not confirmed)
**Code:** https://github.com/ParaMind2025/isoquant
**Note:** Missed in all prior collects; late-caught on April 22.

---

## Context: Rotation-Family KV Compression

TurboQuant (ICLR 2026, arXiv 2504.19874) introduced the idea of applying an orthogonal rotation to KV vectors before scalar quantization. The rotation decorrelates channels and spreads outliers, improving Lloyd-Max quantization quality. TurboQuant uses a dense d×d random orthogonal matrix Π — for d=128, this requires 16,384 multiply-adds per vector.

RotorQuant (2026, not peer-reviewed; code: github.com/scrya-com/rotorquant) replaced Π with Clifford rotors in Cl(3,0): the rotor sandwich product R x R̃ uses only ~100 multiply-adds per vector (exploiting algebraic sparsity). RotorQuant achieves 10–19× speedup vs TurboQuant on NVIDIA, 9–31× on Apple Silicon, with 44× fewer parameters, while matching attention cosine similarity (0.990 vs TurboQuant's 0.991 at d=128).

IsoQuant is the next step: it shows that RotorQuant's 3D Cl(3,0) partition is poorly aligned with modern hardware (not power-of-2 block sizes, limited local mixing), and replaces it with isoclinic decomposition of SO(4).

---

## Key Technical Contribution

### Isoclinic Decomposition of SO(4)

Any rotation in SO(4) can be written as the product of two isoclinic rotations: a left-isoclinic factor (q_L-multiplication) and a right-isoclinic factor (q_R-multiplication). In quaternion notation:

```
T(v) = q_L ⊗ v ⊗ q̄_R
```

where v is a 4D vector interpreted as a quaternion, q_L and q_R are unit quaternions (left and right isoclinic factors), and ⊗ is quaternion multiplication.

This is hardware-aligned because:
1. Each quaternion multiplication is a fixed-size 4-wide operation — maps directly to SIMD width
2. Power-of-2 block size (4D) enables coalesced memory access on GPU
3. No algebraic sparsity required — the isoclinic structure guarantees efficiency by construction

### Variants

- **IsoQuant-Full**: applies both left and right isoclinic factors — full SO(4) rotation; costs 1,024 FMAs at d=128
- **IsoQuant-Fast**: keeps only one isoclinic factor — lower cost; costs 512 FMAs at d=128
- **IsoQuant-2D**: lightweight special case for 2D blocks

### Comparison at d=128

| Method | FMAs per vector | Relative cost |
|--------|-----------------|---------------|
| TurboQuant (dense d×d) | 16,384 | 1× (baseline) |
| RotorQuant (Cl(3,0)) | ~2,408 | ~0.15× |
| IsoQuant-Full | 1,024 | ~0.06× |
| IsoQuant-Fast | 512 | ~0.03× |

---

## Performance Results

Tested across 18 settings (combinations of model, context length, bit width):

- **IsoQuant-Full**: average **4.49× speedup** over RotorQuant fused CUDA
- **IsoQuant-Fast**: average **4.66× speedup** over RotorQuant fused CUDA

Reconstruction quality (cosine similarity, attention accuracy): **essentially identical** to RotorQuant and TurboQuant across all 18 settings.

---

## Relationship to Prior Work

- **TurboQuant**: conceptual reference for dense rotation; IsoQuant does not benchmark directly vs TurboQuant runtime (different kernel implementations)
- **RotorQuant**: direct runtime baseline; IsoQuant-Full is 4.49× faster than RotorQuant
- **Stage-2 estimators (Lloyd-Max)**: IsoQuant is orthogonal to the quantization step — it reduces rotation cost while remaining compatible with any quantizer applied after rotation

---

## vLLM Status

No vLLM integration. A vLLM feature request exists for RotorQuant support (issue #38291) but is not merged. IsoQuant is research-only as of April 2026.

---

## Lineage Summary

```
TurboQuant (dense d×d, ICLR 2026)
  → RotorQuant (Clifford rotors Cl(3,0), 2026)
    → IsoQuant (SO(4) isoclinic, March 2026)
```

Each step reduces rotation computational cost while preserving compression quality.
