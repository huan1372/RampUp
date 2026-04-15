# Karpathy LLM Knowledge Bases (LLM Wiki)
- Source: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Ingested: 2026-04-14
- Type: Gist / methodology document
- Published: April 4, 2026

Core idea:
Instead of RAG (re-deriving knowledge every query), the LLM incrementally builds and maintains
a persistent wiki — structured, interlinked markdown files. The wiki is a compounding artifact.

Architecture (3 layers):
1. Raw sources — immutable source documents (articles, papers, data). Source of truth.
2. The wiki — LLM-generated markdown. Summaries, entity pages, concept pages, cross-links.
   The LLM owns this entirely. Human reads, LLM writes.
3. The schema — configuration (CLAUDE.md / AGENTS.md) that tells the LLM how the wiki works.

Operations:
- Ingest: new source → LLM reads, extracts, integrates into wiki, updates links
- Query: LLM reads index → follows links → synthesizes answer → notes gaps
- Lint: health checks for contradictions, broken links, stale pages
- Maintenance: wiki evolves with every source and every question

Key quotes from community:
- "the wiki is a persistent, compounding artifact"
- Scale: ~100 articles, ~400K words — no vector DB needed at this scale
- Tools: Obsidian as frontend, Claude Code / OpenClaw as the agent
- Output formats: markdown files, Marp slides, matplotlib charts
