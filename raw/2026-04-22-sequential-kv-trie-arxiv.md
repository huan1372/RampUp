---
title: "Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.15356
collected: 2026-04-22
tags: [kv-cache, compression, quantization, information-theory, sequential-compression, prefix-caching]
---

# Sequential KV Cache Compression via Probabilistic Language Tries

**arXiv:** 2604.15356
**Submitted:** April 10, 2026
**Authors:** Gregory Magarshak
**Note:** Deferred from April 21 collect; technical content confirmed on April 22.

---

## Core Claim

All existing per-vector KV cache compression methods — including TurboQuant, FP8, and scalar quantization — are bounded by the Shannon entropy of individual KV vectors. This paper proves that the KV cache as a sequence can be compressed below this per-vector limit by exploiting sequential structure.

**Key insight:** KV cache tokens are not arbitrary floating-point data — they are samples from the exact formal language the model was trained on. The model is by construction a near-optimal predictor of that language. This means the model itself can predict what each successive KV vector will be, and only the residual (the surprise) needs to be stored.

Unlike per-vector methods whose compression ratio is fixed by head dimension, sequential compression improves with context length because a model that has processed more tokens has a more precise predictive distribution.

---

## Two-Layer Architecture

### Layer 1: Probabilistic Prefix Deduplication

Uses the Probabilistic Language Trie (PLT) metric — a trie-based framework that assigns semantic distance to token sequences — to identify shared prefixes across sessions that are semantically equivalent. Rather than exact string matching (as in standard prefix caching), PLT deduplication groups probabilistically equivalent prefixes and stores the KV cache for one representative sequence, deduplicating across sessions.

This is a generalization of prefix caching: standard prefix caching deduplicates on exact token match; PLT deduplication deduplicates on semantic equivalence.

### Layer 2: Predictive Delta Coding

After prefix deduplication, each remaining KV vector is compressed by storing only the residual from the model's own prediction of what that KV should be.

Formally: instead of storing KV_t directly, store KV_t − f(context_<t), where f(·) is the model's predicted KV for position t given prior context. Since the model is near-optimal at predicting the next token in its training language, f(·) is close to KV_t, and the residual is small.

---

## Properties

**Shannon limit breakthrough:**
Per-vector entropy H(KV_t) is fixed by the dimensionality and distribution of individual KV vectors. Sequential methods compress at H(KV_t | context), which can be arbitrarily smaller for a good predictor.

**Scaling with context:**
As context length grows, the model's prediction of each successive KV vector becomes more precise (lower H(KV_t | context)), so the residual shrinks and compression ratio improves. Per-vector methods achieve fixed compression regardless of context length.

**Compatibility:**
The second layer (predictive delta coding) is orthogonal to per-vector quantization — delta residuals can themselves be quantized with FP8 or TurboQuant after delta coding.

---

## Limitations / Open Questions

- This is a theoretical / algorithmic paper; no benchmark numbers vs TurboQuant or FP8 in the web-accessible content
- PLT metric computation cost is not characterized; real-time PLT similarity matching across sessions adds system complexity
- Requires model forward pass to compute predicted KV — potential throughput impact for online compression
- No vLLM integration; the paper proposes the framework without implementation artifacts

---

## Relationship to Prior Work

- **TurboQuant / FP8**: operate at the per-vector level; bounded by per-vector Shannon entropy
- **Standard prefix caching (vLLM)**: deduplicates exact token-match prefixes; Layer 1 of this paper generalizes this to semantic equivalence
- **Probabilistic Language Tries (PLT, arXiv 2604.06228)**: foundational framework for the trie metric used in Layer 1
