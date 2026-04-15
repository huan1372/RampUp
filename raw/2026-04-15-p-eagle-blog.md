---
title: "P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM"
source_url: https://vllm.ai/blog/p-eagle
collected: 2026-04-15
published: 2026-03-13
type: blog-post
author: vLLM Team
---

# P-EAGLE: Parallel Speculative Decoding

## Overview
P-EAGLE transforms EAGLE's autoregressive drafting into parallel generation. Instead of generating draft tokens one-by-one, all K draft tokens are produced in a single forward pass. Available in vLLM v0.16.0+.

## How It Works

### Step 1 — Prefilling
Target model processes the prompt and generates a token while capturing internal hidden states at each position:
- `h_prompt` — hidden states at prompt positions
- `h_context` — hidden states at newly generated token positions

### Step 2 — P-EAGLE Drafter
Drafter constructs inputs for all positions simultaneously:
- **Position 1** (Next-Token-Prediction): pairs new token embedding with `h_context`
- **Positions 2–K** (Multi-Token-Prediction): uses learnable "mask" parameters as placeholders for unknown future tokens

All K draft tokens produced in **a single forward pass** (vs. K sequential passes in EAGLE).

### Implementation
- Pre-trained models: GPT-OSS 120B, GPT-OSS 20B, Qwen3-Coder 30B
- Lightweight 4-layer drafter model
- Fused Triton kernel handles batch metadata reconstruction on-GPU

## Performance Results

### Speedup over EAGLE-3
- Max speedup: **1.69× on SPEED-Bench** (concurrency=1)
- MT-Bench: 1.55× (c=1) → 1.05× (c=64) — benefit diminishes at high concurrency
- HumanEval: 1.55× (c=1) → 1.23× (c=64)

### Acceptance Length (AL) Comparison

| Benchmark    | Method    | K=3  | K=7  |
|--------------|-----------|------|------|
| HumanEval    | P-EAGLE   | 3.02 | 3.94 |
| HumanEval    | EAGLE-3   | 2.65 | 3.03 |
| SPEED-Bench  | P-EAGLE   | 2.87 | 3.38 |
| SPEED-Bench  | EAGLE-3   | 2.24 | 2.59 |

At K=7: P-EAGLE shows ~30-31% higher acceptance rates on both benchmarks.

### Optimal Speculation Depth
- P-EAGLE: peak throughput at K=7
- EAGLE-3: maxes out at K=3
- Reason: single-pass drafting eliminates the overhead that limits K in EAGLE-3

## Key Insight
By replacing autoregressive draft generation with a single parallel forward pass, P-EAGLE makes larger speculation depths (K=7 vs K=3) economical. The mask parameters for unknown future positions are learned during training.
