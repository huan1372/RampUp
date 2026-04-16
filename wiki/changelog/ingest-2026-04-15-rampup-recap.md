---
title: "Ingest Run — 2026-04-15 (ramp-up recap PDF)"
tags: [changelog, ingest, manual]
created: 2026-04-15
updated: 2026-04-15
sources: [raw/2026-04-14-vllm-rampup-recap.md]
related: [wiki/index.md, concepts/paged-attention.md, concepts/continuous-batching.md, concepts/chunked-prefill.md, overview.md]
---

# Manual Ingest — vLLM Ramp-Up Recap

## What was ingested
`raw/2026-04-14-vllm-rampup-recap.md` — markdown conversion of `vllm-recap.pdf`,
a Q&A session summary covering vLLM fundamentals: prefill/decode pipeline,
PagedAttention, continuous batching, chunked prefill, GPU memory layout,
key parameters, competitive benchmarks, and glossary.

## Pages updated

| Page | What changed |
|---|---|
| [concepts/paged-attention.md](../concepts/paged-attention.md) | Added "Statistical Multiplexing" section; added 60-80% baseline waste figure |
| [concepts/continuous-batching.md](../concepts/continuous-batching.md) | Added static-vs-continuous timeline diagram; clarified "tightly coupled" with PagedAttention; reframed as "streaming admission" |
| [concepts/chunked-prefill.md](../concepts/chunked-prefill.md) | Added fairness framing + head-of-line blocking; added "Why Prefill Needs Special Handling" section with compute-bound vs memory-bandwidth-bound contrast; TTFT/ITL targets |
| [overview.md](../overview.md) | Added Glossary (TTFT, ITL, KV cache, Prefill, Decode, Preemption, TP) |
| [index.md](../index.md) | Added link to this changelog entry |

## No new pages created
All concepts in the recap already had wiki coverage. The ingest enriched existing
pages rather than expanding the topic graph.

## Key new claims / insights added

- **Statistical multiplexing** as the underlying principle of PagedAttention — the shared pool can be *smaller* than the sum of per-request worst-case reservations
- **"Tightly coupled"** — continuous batching was architecturally impossible before PagedAttention because contiguous memory could not be partially freed
- **Fairness framing** of chunked prefill — slightly penalize one heavy request to unblock dozens of lighter ones
- **Hard latency targets** — TTFT < 200ms, ITL < 30ms for fluid streaming
- **Prefill vs decode resource profile** — compute-bound vs memory-bandwidth-bound

## Contradictions / conflicts
None found. The recap is consistent with existing wiki content and adds colour rather
than contradicting.

## Lint results
- All cross-links resolve ✓
- All claims cite a source in `raw/` ✓
- All updated pages have current `updated` date ✓
