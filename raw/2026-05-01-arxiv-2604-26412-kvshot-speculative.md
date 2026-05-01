---
title: "When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.26412
collected: 2026-05-01
tags: [speculative-decoding, kv-cache, long-context, drafting, training]
---

# When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?

**arXiv:** 2604.26412  
**Submitted:** Late April 2026

## Problem: Long-Range Draft Accuracy Decay

Hidden-state-based speculative decoding drafters (EAGLE, EAGLE-3, Medusa) generate K draft tokens by using the target model's last hidden state as a compressed context summary. This summary biases toward information relevant to the *current* query position — but at longer speculative steps (step k >> 1), the summary becomes increasingly misaligned with what information the drafter actually needs.

The result is **long-range draft accuracy decay**: acceptance rate decreases as the speculative step index increases. This decay persists even in test-time training (TTT)-tuned drafters, suggesting the root cause is architectural, not just a training data issue.

## KVShot: Diagnostic Framework

The authors introduce **KVShot**, a diagnostic framework that systematically compares three KV/hidden-state reuse paradigms:

1. **Hidden-only (baseline)**: drafter receives only the target hidden state; standard EAGLE-family design
2. **KV-only**: drafter receives the target model's KV cache directly, not the hidden state; the drafter's attention layers can attend to the full token-level KV history
3. **Hybrid**: drafter receives both KV cache and hidden state

**Test model**: Qwen3-8B (target); drafter architecture not specified (likely a reduced version)

## Key Findings

**KV-Reuse helps long-range acceptance:**  
Providing the target model's KV cache to the drafter improves acceptance at high speculative step indices (k > 3). The KV cache's explicit per-token representations preserve context better than the collapsed hidden-state summary, reducing the long-range drift problem.

**But end-to-end speedups remain marginal:**  
Despite the improved acceptance, the wall-clock end-to-end speedup from KV-aware decoding is marginal under current training pipelines. The theoretical acceptance rate gain does not translate proportionally to tokens-per-second improvement.

**Two structural bottlenecks identified:**

1. **Shallow drafter architecture**: small drafters cannot accurately estimate target model queries from the provided KV cache — the key projection from the target model's perspective requires depth to compute accurately. With a shallow drafter, the KV cache is available but cannot be fully leveraged.

2. **Sparse gradient signals for KV projections**: during drafter training, the draft-side KV projections (the layers that learn to use the target's KV cache) receive sparse gradient signals because they are only activated at positions where KV reuse actually differs from hidden-state-only paths. Sparse gradients → poor learning of the reuse mechanism.

## Proposed Direction

The bottlenecks suggest that **block-wise training paradigms** are needed to fully realize KV-aware speculative decoding:

- Train the drafter in blocks that explicitly force gradient signal through KV projection layers (not just the final-step acceptance signal)
- This is contrasted with TTT (Test-Time Training), which the paper shows is insufficient — TTT improves accuracy marginally but does not fix the sparse-gradient problem for KV projections

## Implications for Existing KB Content

- **EAGLE-3 / P-EAGLE**: both are hidden-state-based drafters. KVShot suggests their long-range acceptance decay is a structural limitation of the hidden-state compression, not a training deficiency. Providing target KV cache could improve EAGLE-3/P-EAGLE but requires architectural changes.
- **CSD (arXiv 2604.13634)**: addresses false rejections (semantically valid draft tokens rejected for lexical divergence); KVShot addresses a different issue — acceptance rate decay over speculative step indices. Complementary diagnoses.
- **SMC-SD (arXiv 2604.15672)**: replaces rejection sampling with importance weighting; does not address the root cause of long-range draft quality degradation. Complementary approach.
- **P-EAGLE**: trains with a single-pass parallel draft — different mechanism but same hidden-state compression problem at long K.

## vLLM Integration

No vLLM integration. KVShot is a diagnostic framework, not a deployable system. The proposed block-wise training direction would require re-training EAGLE/P-EAGLE-class drafters with a new loss formulation — a significant research-to-production gap.
