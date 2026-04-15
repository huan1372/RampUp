# Inference Optimization Knowledge Base — Schema

This is a personal knowledge base focused on **LLM inference optimization**, following the [Karpathy LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). The LLM maintains the wiki; the human curates raw sources and asks questions.

## Architecture

```
raw/                    ← immutable source documents (articles, papers, commits, notes)
wiki/
  ├── index.md          ← master index of all wiki pages (auto-maintained)
  ├── overview.md       ← high-level synthesis of the entire knowledge base
  ├── concepts/         ← compiled concept articles (one per concept)
  ├── techniques/       ← optimization techniques with benchmarks
  ├── changelog/        ← weekly digests and learning summaries
  └── _templates/       ← templates for each page type
```

## Conventions

### File naming
- Lowercase, hyphen-separated: `paged-attention.md`, `chunked-prefill.md`
- Concept pages: `wiki/concepts/<name>.md`
- Technique pages: `wiki/techniques/<name>.md`
- Weekly digests: `wiki/changelog/week-YYYY-MM-DD.md`

### Page structure
Every wiki page MUST include YAML frontmatter:

```yaml
---
title: "<Page Title>"
tags: [tag1, tag2, tag3]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [raw/<filename1>, raw/<filename2>]
related: [wiki/concepts/<page>, wiki/techniques/<page>]
---
```

### Cross-linking
- Use standard markdown links: `[PagedAttention](../concepts/paged-attention.md)`
- Every page must link to at least one related page
- When creating a new page, update `index.md` and all related pages

### Source attribution
- Every claim must trace back to a file in `raw/`
- Use inline citations: `(source: raw/vllm-pagedattention-paper.md)`

## Operations

### INGEST — when a new source is added to `raw/`
1. Read the new source completely
2. Identify key concepts, techniques, benchmarks, and claims
3. For each concept/technique:
   - If a wiki page exists → update it with new information, note contradictions
   - If no page exists → create one using the appropriate template
4. Update `wiki/index.md` with any new pages
5. Update `wiki/overview.md` if the new source changes the big picture
6. Add cross-links between new and existing pages

### QUERY — when the user asks a question
1. Read `wiki/index.md` to find relevant pages
2. Read those pages and follow cross-links as needed
3. Synthesize an answer grounded in wiki content
4. If the answer reveals a gap, note it in the relevant page under `## Open Questions`

### LINT — periodic health check
1. Scan all wiki pages for:
   - Broken cross-links
   - Pages with no `sources` in frontmatter
   - Stale information (updated date > 30 days ago)
   - Contradictions between pages
2. Report findings and fix what can be auto-fixed
3. Log the lint run in `wiki/changelog/`

### DIGEST — weekly summary
1. Collect all pages updated in the past 7 days
2. Summarize what was learned, what changed, what contradictions emerged
3. Write to `wiki/changelog/week-YYYY-MM-DD.md`
4. Highlight open questions and suggested next reads

## Scope

This knowledge base covers:
- **vLLM internals**: PagedAttention, continuous batching, scheduling, KV cache management, Model Runner V2
- **Inference optimization**: quantization (FP8, FP4, GPTQ, AWQ), speculative decoding, tensor parallelism, pipeline parallelism, expert parallelism
- **Serving infrastructure**: disaggregated prefill/decode, prefill-decode scheduling, GPU memory management
- **Benchmarking**: throughput, TTFT, ITL, P99 latency, tools (GuideLLM, vLLM benchmarks)
- **Hardware**: NVIDIA (H100, H200, B200, GB200 Blackwell), AMD (MI300X, MI350X), TPU
- **Comparison engines**: vLLM vs SGLang vs TensorRT-LLM
- **Agent patterns**: tool use, skill architecture, cron jobs, subagents (relevant to OpenClaw integration)
